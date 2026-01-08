"""
Fractal Probability Put Writing Strategy
=========================================
Modelo de estimativa de probabilidade de exercício baseado em Geometria Fractal.
Compara a probabilidade Black-Scholes tradicional com o Movimento Browniano Fracionário.
"""

import io
import zipfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bcb import sgs
from scipy.stats import norm, linregress


# =============================================================================
# DATA CAPTURE FUNCTIONS
# =============================================================================

def download_b3_option_data(ticker: str, date: str) -> dict:
    """
    Downloads option data from B3 and extracts last price and strike.
    
    Args:
        ticker: Option ticker (e.g., 'PRION410')
        date: Date in format 'YYYY-MM-DD'
    
    Returns:
        dict with 'last_price' and 'strike'
    """
    url = f"https://arquivos.b3.com.br/rapinegocios/tickercsv/{ticker}/{date}"
    
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    # Extract ZIP contents
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        csv_filename = z.namelist()[0]
        with z.open(csv_filename) as f:
            df = pd.read_csv(f, sep=';', encoding='latin-1')
    
    # Parse relevant columns
    # B3 CSV columns: PrecoUltimo (or similar) for last price
    # The strike is typically encoded in the ticker or in a specific column
    
    # Get last traded price
    if 'PreçoNegócio' in df.columns:
        last_price = df['PreçoNegócio'].iloc[-1]
    elif 'PrecoNegocio' in df.columns:
        last_price = df['PrecoNegocio'].iloc[-1]
    else:
        # Try to find price column
        price_cols = [c for c in df.columns if 'preco' in c.lower() or 'preço' in c.lower()]
        if price_cols:
            last_price = df[price_cols[0]].iloc[-1]
        else:
            last_price = df.iloc[-1, -1]  # Last column, last row as fallback
    
    # Convert to float if string
    if isinstance(last_price, str):
        last_price = float(last_price.replace(',', '.'))
    
    # Extract strike from ticker (e.g., PRION410 -> 41.0)
    # Format: [BASE][MONTH_CODE][STRIKE]
    # N = December Put, strike 410 = 41.0
    strike_str = ticker[-3:]  # Last 3 digits
    strike = float(strike_str) / 10  # 410 -> 41.0
    
    return {
        'ticker': ticker,
        'last_price': float(last_price),
        'strike': strike,
        'date': date
    }


def get_selic_rate() -> float:
    """
    Fetches current Selic target rate from BCB (SGS code 432).
    
    Returns:
        Annual Selic rate as decimal (e.g., 0.1175 for 11.75%)
    """
    # SGS code 432 = Selic Meta (annual rate)
    selic = sgs.get({'selic': 432}, last=1)
    return selic['selic'].iloc[-1] / 100


def get_stock_history(ticker: str, days: int = 252) -> pd.Series:
    """
    Gets historical closing prices for the underlying asset.
    
    Args:
        ticker: Stock ticker (e.g., 'PRIO3.SA')
        days: Number of trading days to fetch
    
    Returns:
        Series of closing prices
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(days * 1.5))  # Buffer for weekends/holidays
    
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date.strftime('%Y-%m-%d'), 
                        end=end_date.strftime('%Y-%m-%d'))
    
    return hist['Close'].tail(days)


# =============================================================================
# HURST EXPONENT CALCULATION (R/S ANALYSIS)
# =============================================================================

def calculate_hurst_exponent(prices: pd.Series) -> float:
    """
    Calculates the Hurst Exponent using Rescaled Range (R/S) Analysis.
    
    H < 0.5: Mean reversion (anti-persistent)
    H = 0.5: Random walk (Brownian motion)
    H > 0.5: Trend persistence (persistent)
    
    Args:
        prices: Series of prices
    
    Returns:
        Hurst exponent (0 to 1)
    """
    # Calculate log returns
    log_returns = np.log(prices / prices.shift(1)).dropna().values
    n = len(log_returns)
    
    if n < 20:
        raise ValueError("Need at least 20 data points for Hurst calculation")
    
    # Calculate R/S for different time scales
    rs_list = []
    n_list = []
    
    # Use divisors of n for clean subseries
    for div in range(10, n // 2):
        if n % div == 0 or True:  # Allow non-exact divisions
            subseries_len = div
            num_subseries = n // subseries_len
            
            if num_subseries < 1:
                continue
            
            rs_values = []
            
            for i in range(num_subseries):
                start_idx = i * subseries_len
                end_idx = start_idx + subseries_len
                subseries = log_returns[start_idx:end_idx]
                
                if len(subseries) < 2:
                    continue
                
                # Mean-adjusted cumulative deviation
                mean_adj = subseries - np.mean(subseries)
                cumsum = np.cumsum(mean_adj)
                
                # Range (R)
                R = np.max(cumsum) - np.min(cumsum)
                
                # Standard deviation (S)
                S = np.std(subseries, ddof=1)
                
                if S > 0:
                    rs_values.append(R / S)
            
            if rs_values:
                rs_list.append(np.mean(rs_values))
                n_list.append(subseries_len)
    
    if len(rs_list) < 3:
        return 0.5  # Default to random walk if insufficient data
    
    # Log-log regression to find Hurst exponent
    log_n = np.log(n_list)
    log_rs = np.log(rs_list)
    
    # Linear regression: log(R/S) = H * log(n) + c
    slope, _ = np.polyfit(log_n, log_rs, 1)
    
    # Bound H to valid range
    hurst = max(0.01, min(0.99, slope))
    
    return hurst


# =============================================================================
# FRACTIONAL BROWNIAN MOTION MONTE CARLO
# =============================================================================

def generate_fbm_paths(S0: float, mu: float, sigma: float, H: float, 
                       T: float, n_steps: int, n_paths: int) -> np.ndarray:
    """
    Generate Fractional Brownian Motion price paths using Cholesky decomposition.
    
    The key difference from standard Brownian motion:
    - Standard BM: increments are independent (H=0.5)
    - Fractional BM: increments are correlated based on H
      - H > 0.5: positive correlation (persistence)
      - H < 0.5: negative correlation (mean reversion)
    
    Args:
        S0: Initial spot price
        mu: Drift (annualized)
        sigma: Volatility (annualized)
        H: Hurst exponent
        T: Time horizon (years)
        n_steps: Number of time steps
        n_paths: Number of simulation paths
    
    Returns:
        Array of shape (n_paths, n_steps+1) with price paths
    """
    dt = T / n_steps
    
    # Build covariance matrix for fBm VALUES (not increments)
    # Cov(B_H(t), B_H(s)) = 0.5 * (|t|^2H + |s|^2H - |t-s|^2H)
    t = np.arange(1, n_steps + 1) * dt
    
    # Covariance matrix for fBm
    cov = np.zeros((n_steps, n_steps))
    for i in range(n_steps):
        for j in range(n_steps):
            ti, tj = t[i], t[j]
            cov[i, j] = 0.5 * (ti**(2*H) + tj**(2*H) - abs(ti - tj)**(2*H))
    
    # Add small regularization for numerical stability
    cov += np.eye(n_steps) * 1e-10
    
    # Cholesky decomposition
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        # Fallback to regular Brownian motion if Cholesky fails
        L = np.eye(n_steps) * np.sqrt(dt)
    
    # Generate fBm values at each time step
    Z = np.random.standard_normal((n_paths, n_steps))
    fBm_values = Z @ L.T  # These are B_H(t_1), B_H(t_2), ..., B_H(t_n)
    
    # Calculate increments: dB_H(t_i) = B_H(t_i) - B_H(t_{i-1})
    fBm_increments = np.zeros_like(fBm_values)
    fBm_increments[:, 0] = fBm_values[:, 0]  # First increment = first value
    fBm_increments[:, 1:] = np.diff(fBm_values, axis=1)  # Subsequent increments
    
    # Generate price paths using geometric fBm
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    
    for i in range(n_steps):
        paths[:, i+1] = paths[:, i] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * fBm_increments[:, i]
        )
    
    return paths


def run_monte_carlo_fbm(S0: float, K: float, r: float, sigma: float, 
                        H: float, T: float, n_paths: int = 5000) -> dict:
    """
    Run Monte Carlo simulation using Fractional Brownian Motion.
    
    Args:
        S0: Current spot price
        K: Strike price
        r: Risk-free rate
        sigma: Volatility
        H: Hurst exponent
        T: Time to expiration (years)
        n_paths: Number of simulation paths
    
    Returns:
        dict with simulation results
    """
    n_days = max(int(T * 365), 1)
    
    # Generate paths
    paths = generate_fbm_paths(S0, r, sigma, H, T, n_days, n_paths)
    
    # Final prices
    final_prices = paths[:, -1]
    
    # Probability of exercise (final price < strike)
    prob_exercise = np.mean(final_prices < K)
    
    # Risk of Ruin: price touches 10% below strike at any point
    ruin_level = K * 0.90
    min_prices = np.min(paths, axis=1)
    prob_ruin = np.mean(min_prices < ruin_level)
    
    # Additional statistics
    avg_final = np.mean(final_prices)
    std_final = np.std(final_prices)
    percentile_5 = np.percentile(final_prices, 5)
    percentile_95 = np.percentile(final_prices, 95)
    
    return {
        'n_paths': n_paths,
        'n_days': n_days,
        'prob_exercise': prob_exercise,
        'prob_ruin': prob_ruin,
        'avg_final': avg_final,
        'std_final': std_final,
        'percentile_5': percentile_5,
        'percentile_95': percentile_95,
        'ruin_level': ruin_level
    }


# =============================================================================
# PROBABILITY MODELS
# =============================================================================

def calculate_d2_bs(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate d2 for Black-Scholes model.
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate (annual)
        sigma: Volatility (annual)
    
    Returns:
        d2 value
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d2


def prob_exercise_bs(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate probability of exercise using traditional Black-Scholes (N(d2)).
    This is the risk-neutral probability that S > K at expiration for a call,
    or equivalently 1 - N(d2) for a put.
    
    For a PUT: probability of exercise = N(-d2) = probability that S < K
    
    Returns:
        Probability of PUT exercise (0 to 1)
    """
    d2 = calculate_d2_bs(S, K, T, r, sigma)
    # For PUT: P(exercise) = P(S < K) = N(-d2)
    return norm.cdf(-d2)


def prob_exercise_fractal(S: float, K: float, T: float, r: float, 
                          sigma: float, H: float) -> float:
    """
    Calculate probability of exercise using Fractional Brownian Motion.
    
    Key difference: Uses sigma * T^H instead of sigma * sqrt(T)
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate (annual)
        sigma: Volatility (annual)
        H: Hurst exponent
    
    Returns:
        Probability of PUT exercise (0 to 1)
    """
    # Fractal volatility scaling
    sigma_fractal = sigma * (T ** H)
    
    # Modified d1 and d2
    d1 = (np.log(S / K) + r * T + 0.5 * sigma_fractal**2) / sigma_fractal
    d2 = d1 - sigma_fractal
    
    # For PUT: P(exercise) = P(S < K) = N(-d2)
    return norm.cdf(-d2)


def calculate_historical_volatility(prices: pd.Series) -> float:
    """
    Calculate annualized historical volatility from prices.
    
    Returns:
        Annualized volatility as decimal
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    daily_vol = log_returns.std()
    annual_vol = daily_vol * np.sqrt(252)
    return annual_vol


def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float) -> dict:
    """
    Calculate option Greeks for a PUT option.
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate (annual)
        sigma: Volatility (annual)
    
    Returns:
        dict with Delta, Gamma, Theta, Vega, Rho
    """
    sqrt_T = np.sqrt(T)
    
    # d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    # Standard normal PDF and CDF
    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    cdf_neg_d1 = norm.cdf(-d1)
    cdf_neg_d2 = norm.cdf(-d2)
    
    # PUT Greeks
    delta = cdf_d1 - 1  # Put Delta is negative (between -1 and 0)
    gamma = pdf_d1 / (S * sigma * sqrt_T)
    theta = (-(S * pdf_d1 * sigma) / (2 * sqrt_T) 
             + r * K * np.exp(-r * T) * cdf_neg_d2) / 365  # Daily theta
    vega = S * pdf_d1 * sqrt_T / 100  # Per 1% change in IV
    rho = -K * T * np.exp(-r * T) * cdf_neg_d2 / 100  # Per 1% change in rate
    
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }


# =============================================================================
# TREND FILTERS
# =============================================================================

def check_trend_filters(prices: pd.Series) -> dict:
    """
    Check 3 trend filters for trading decision.
    
    Args:
        prices: Series of closing prices
    
    Returns:
        dict with filter results and values
    """
    current_price = prices.iloc[-1]
    
    # Filter A: Price > SMA 21
    sma_21 = prices.tail(21).mean()
    filter_a = current_price > sma_21
    
    # Filter B: 30-day Momentum > 0
    if len(prices) >= 30:
        momentum_30 = (current_price / prices.iloc[-30] - 1) * 100
    else:
        momentum_30 = (current_price / prices.iloc[0] - 1) * 100
    filter_b = momentum_30 > 0
    
    # Filter C: 30-day Linear Regression Slope > 0
    prices_30 = prices.tail(30)
    x = np.arange(len(prices_30))
    y = prices_30.values
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    filter_c = slope > 0
    
    return {
        'filter_a': filter_a,  # Price > SMA21
        'filter_b': filter_b,  # Momentum 30d > 0
        'filter_c': filter_c,  # Slope > 0
        'sma_21': sma_21,
        'momentum_30': momentum_30,
        'slope': slope,
        'r_squared': r_value ** 2,
        'all_bullish': filter_a and filter_b and filter_c
    }


def get_recommendation(hurst: float, filters: dict, spot: float) -> tuple:
    """
    Generate trading recommendation based on Hurst and trend filters.
    
    Returns:
        tuple of (classification, recommendation_text, risk_level)
    """
    sma_21 = filters['sma_21']
    
    # Cenario PERSISTENTE (H > 0.55)
    if hurst > 0.55:
        if filters['all_bullish']:
            return (
                "VENDA FORTE",
                "Tendencia de alta confirmada e persistente. Baixo risco de exercicio.",
                "LOW"
            )
        else:
            return (
                "RISCO ALTO",
                "Mercado persistente em queda ou sem direcao clara. Perigo de movimento forte contra a Put.",
                "HIGH"
            )
    
    # Cenario REVERSAO (H < 0.45)
    elif hurst < 0.45:
        if spot < sma_21:
            return (
                "OPORTUNIDADE",
                "Ativo esticado para baixo com caracteristica de reversao. Boa probabilidade de retorno ao strike.",
                "LOW"
            )
        else:
            return (
                "CAUTELA",
                "Ativo em exaustao de topo com tendencia de voltar para a media.",
                "MEDIUM"
            )
    
    # Cenario RUIDO (0.45 <= H <= 0.55)
    else:
        return (
            "NEUTRO",
            "Siga estritamente as probabilidades do Delta de Black-Scholes tradicional.",
            "MEDIUM"
        )


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_put_writing(option_ticker: str, option_date: str, 
                        underlying_ticker: str, expiration_date: str) -> pd.DataFrame:
    """
    Main analysis function for fractal put writing strategy.
    
    Args:
        option_ticker: B3 option ticker (e.g., 'PRION410')
        option_date: Date for option data (YYYY-MM-DD)
        underlying_ticker: Stock ticker (e.g., 'PRIO3.SA')
        expiration_date: Option expiration date (YYYY-MM-DD)
    
    Returns:
        DataFrame with analysis results
    """
    print("=" * 60)
    print("FRACTAL PUT WRITING ANALYSIS")
    print("=" * 60)
    
    # 1. Capture option data
    print("\n[1] Fetching option data from B3...")
    option_data = download_b3_option_data(option_ticker, option_date)
    print(f"   Ticker: {option_data['ticker']}")
    print(f"   Last Price: R$ {option_data['last_price']:.2f}")
    print(f"   Strike: R$ {option_data['strike']:.2f}")
    
    # 2. Get Selic rate
    print("\n[2] Fetching Selic rate from BCB...")
    selic = get_selic_rate()
    print(f"   Selic: {selic * 100:.2f}%")
    
    # 3. Get stock history
    print(f"\n[3] Fetching {underlying_ticker} history...")
    prices = get_stock_history(underlying_ticker, days=252)
    spot = prices.iloc[-1]
    print(f"   Current Spot: R$ {spot:.2f}")
    print(f"   Data points: {len(prices)}")
    
    # 4. Calculate volatility
    sigma = calculate_historical_volatility(prices)
    print(f"   Historical Volatility: {sigma * 100:.2f}%")
    
    # 5. Calculate Hurst exponent
    print("\n[4] Calculating Hurst Exponent (R/S Analysis)...")
    hurst = calculate_hurst_exponent(prices)
    print(f"   Hurst Exponent: {hurst:.4f}")
    
    # Calculate recent momentum (last 20 days vs previous 20 days)
    recent_return = (prices.iloc[-1] / prices.iloc[-20] - 1) * 100
    
    # Interpret Hurst + Direction
    if hurst < 0.45:
        hurst_interpretation = "MEAN REVERSION"
        trend_direction = "Reversal expected"
    elif hurst > 0.55:
        hurst_interpretation = "TREND PERSISTENCE"
        if recent_return > 0:
            trend_direction = "BULLISH (alta tende a continuar)"
        else:
            trend_direction = "BEARISH (queda tende a continuar)"
    else:
        hurst_interpretation = "RANDOM WALK"
        trend_direction = "No clear direction"
    
    print(f"   Interpretation: {hurst_interpretation}")
    print(f"   Recent 20d Return: {recent_return:+.2f}%")
    print(f"   Direction: {trend_direction}")
    
    # 6. Calculate time to expiration
    exp_date = datetime.strptime(expiration_date, '%Y-%m-%d')
    current_date = datetime.strptime(option_date, '%Y-%m-%d')
    T = (exp_date - current_date).days / 365
    print(f"\n[5] Time to expiration: {T * 365:.0f} days ({T:.4f} years)")
    
    # 7. Calculate probabilities
    print("\n[6] Calculating exercise probabilities...")
    prob_bs = prob_exercise_bs(spot, option_data['strike'], T, selic, sigma)
    prob_fractal = prob_exercise_fractal(spot, option_data['strike'], T, selic, sigma, hurst)
    
    print(f"   Black-Scholes (N(-d2)): {prob_bs * 100:.2f}%")
    print(f"   Fractal (T^H):          {prob_fractal * 100:.2f}%")
    print(f"   Difference:             {(prob_bs - prob_fractal) * 100:+.2f} p.p.")
    
    # 7b. Calculate Greeks
    print("\n[6b] Calculating Greeks (PUT)...")
    greeks = calculate_greeks(spot, option_data['strike'], T, selic, sigma)
    
    print(f"   Delta: {greeks['delta']:.4f}")
    print(f"   Gamma: {greeks['gamma']:.4f}")
    print(f"   Theta: R$ {greeks['theta']:.4f} /dia")
    print(f"   Vega:  R$ {greeks['vega']:.4f} /1% IV")
    print(f"   Rho:   R$ {greeks['rho']:.4f} /1% taxa")
    
    # 7c. Monte Carlo fBm Simulation
    print("\n[6c] Running Monte Carlo fBm Simulation (5000 paths)...")
    mc_results = run_monte_carlo_fbm(spot, option_data['strike'], selic, sigma, hurst, T, n_paths=5000)
    
    print(f"   Paths simulated: {mc_results['n_paths']}")
    print(f"   Time horizon:    {mc_results['n_days']} days")
    print(f"   Avg final price: R$ {mc_results['avg_final']:.2f}")
    print(f"   Std final price: R$ {mc_results['std_final']:.2f}")
    print(f"   5th percentile:  R$ {mc_results['percentile_5']:.2f}")
    print(f"   95th percentile: R$ {mc_results['percentile_95']:.2f}")
    print(f"")
    print(f"   Prob. Exercicio (MC):     {mc_results['prob_exercise'] * 100:.2f}%")
    print(f"   Prob. Exercicio (BS):     {prob_bs * 100:.2f}%")
    print(f"   Prob. Exercicio (Fractal):{prob_fractal * 100:.2f}%")
    print(f"")
    print(f"   RISCO DE RUINA (-10% do Strike):")
    print(f"   Nivel de Ruina: R$ {mc_results['ruin_level']:.2f}")
    print(f"   Prob. Tocar Ruina: {mc_results['prob_ruin'] * 100:.2f}%")
    
    # 8. Trend Filters
    print("\n[7] Checking Trend Filters...")
    filters = check_trend_filters(prices)
    
    print(f"   Filter A (Price > SMA21): {'YES' if filters['filter_a'] else 'NO'} (SMA21: R$ {filters['sma_21']:.2f})")
    print(f"   Filter B (Momentum 30d):  {'YES' if filters['filter_b'] else 'NO'} ({filters['momentum_30']:+.2f}%)")
    print(f"   Filter C (Slope > 0):     {'YES' if filters['filter_c'] else 'NO'} (slope: {filters['slope']:.4f}, R2: {filters['r_squared']:.2f})")
    print(f"   All Bullish:              {'YES' if filters['all_bullish'] else 'NO'}")
    
    # 9. Trading recommendation
    print("\n" + "=" * 60)
    print("TRADING RECOMMENDATION")
    print("=" * 60)
    
    classification, recommendation, risk_level = get_recommendation(hurst, filters, spot)
    prob_gap = (prob_bs - prob_fractal) * 100
    
    # Risk level indicators
    risk_indicators = {'LOW': '[+]', 'MEDIUM': '[=]', 'HIGH': '[!]'}
    indicator = risk_indicators.get(risk_level, '[?]')
    
    print(f"\n{indicator} {classification}")
    print(f"    {recommendation}")
    print(f"\n    Prob. Exercicio BS:      {prob_bs * 100:.2f}%")
    print(f"    Prob. Exercicio Fractal: {prob_fractal * 100:.2f}%")
    print(f"    GAP BS vs Fractal:       {prob_gap:+.2f} p.p.")
    if prob_gap > 0:
        print("    -> Mercado SUPERESTIMA risco (vantagem para vendedor)")
    elif prob_gap < 0:
        print("    -> Mercado SUBESTIMA risco (desvantagem para vendedor)")
    else:
        print("    -> Modelos convergem")
    
    # 10. Create result DataFrame
    trend_direction = "ALTA" if filters['all_bullish'] else "BAIXA/INDEFINIDA"
    
    result = pd.DataFrame([{
        'Ticker': option_data['ticker'],
        'Strike': option_data['strike'],
        'Spot': round(spot, 2),
        'Hurst': round(hurst, 4),
        'Tendencia': trend_direction,
        'Prob_BS': f"{prob_bs * 100:.2f}%",
        'Prob_Fractal': f"{prob_fractal * 100:.2f}%",
        'GAP': f"{prob_gap:+.2f} p.p.",
        'Recomendacao': classification
    }])
    
    print("\n" + "=" * 60)
    print("EXECUTIVE SUMMARY")
    print("=" * 60)
    print(result.to_string(index=False))
    
    return result


# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Example: PRIO3 Put with strike 41.0 (PRION410)
    # N = December Put
    result_df = analyze_put_writing(
        option_ticker="PRION410",
        option_date="2026-01-07",
        underlying_ticker="PRIO3.SA",
        expiration_date="2026-01-20"  # Third Monday of January
    )
