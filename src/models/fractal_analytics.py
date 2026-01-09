"""
Fractal Analytics Module
========================
Funções de análise fractal para estimativa de probabilidade de exercício
baseada no Expoente de Hurst e Movimento Browniano Fracionário.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, linregress


# =============================================================================
# HURST EXPONENT CALCULATION (R/S ANALYSIS)
# =============================================================================

def calculate_hurst_exponent(prices: pd.Series) -> float:
    """
    Calcula o Expoente de Hurst usando Análise R/S (Rescaled Range).
    
    H < 0.5: Reversão à média (anti-persistente)
    H = 0.5: Random walk (movimento browniano)
    H > 0.5: Tendência persistente
    
    Args:
        prices: Series de preços
    
    Returns:
        Expoente de Hurst (0 a 1)
    """
    # Calcula retornos logarítmicos
    log_returns = np.log(prices / prices.shift(1)).dropna().values
    n = len(log_returns)
    
    if n < 20:
        return 0.5  # Default para random walk se dados insuficientes
    
    # Calcula R/S para diferentes escalas temporais
    rs_list = []
    n_list = []
    
    for div in range(10, n // 2):
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
            
            # Desvio acumulado ajustado pela média
            mean_adj = subseries - np.mean(subseries)
            cumsum = np.cumsum(mean_adj)
            
            # Range (R)
            R = np.max(cumsum) - np.min(cumsum)
            
            # Desvio padrão (S)
            S = np.std(subseries, ddof=1)
            
            if S > 0:
                rs_values.append(R / S)
        
        if rs_values:
            rs_list.append(np.mean(rs_values))
            n_list.append(subseries_len)
    
    if len(rs_list) < 3:
        return 0.5
    
    # Regressão log-log para encontrar H
    log_n = np.log(n_list)
    log_rs = np.log(rs_list)
    
    slope, _ = np.polyfit(log_n, log_rs, 1)
    
    # Limita H ao intervalo válido
    hurst = max(0.01, min(0.99, slope))
    
    return hurst


def get_hurst_interpretation(hurst: float, recent_return: float) -> tuple:
    """
    Interpreta o expoente de Hurst com direção da tendência.
    
    Returns:
        tuple: (interpretation_label, trend_direction, color)
    """
    if hurst < 0.45:
        return ("REVERSÃO À MÉDIA", "Reversão esperada", "#FFB302")
    elif hurst > 0.55:
        if recent_return > 0:
            return ("PERSISTENTE", "Alta tende a continuar", "#39E58C")
        else:
            return ("PERSISTENTE", "Queda tende a continuar", "#FF4B4B")
    else:
        return ("RANDOM WALK", "Sem direção clara", "#636EFA")


# =============================================================================
# PROBABILITY MODELS
# =============================================================================

def calculate_d2_bs(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calcula d2 para o modelo Black-Scholes."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d2


def prob_exercise_bs(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Probabilidade de exercício usando Black-Scholes tradicional (N(-d2)).
    Para PUT: P(exercício) = P(S < K) = N(-d2)
    """
    d2 = calculate_d2_bs(S, K, T, r, sigma)
    return norm.cdf(-d2)


def prob_exercise_fractal(S: float, K: float, T: float, r: float, 
                          sigma: float, H: float) -> float:
    """
    Probabilidade de exercício usando Movimento Browniano Fracionário.
    Diferença chave: usa sigma * T^H ao invés de sigma * sqrt(T)
    """
    # Escala de volatilidade fractal
    sigma_fractal = sigma * (T ** H)
    
    # d1 e d2 modificados
    d1 = (np.log(S / K) + r * T + 0.5 * sigma_fractal**2) / sigma_fractal
    d2 = d1 - sigma_fractal
    
    return norm.cdf(-d2)


def calculate_historical_volatility(prices: pd.Series) -> float:
    """Calcula volatilidade histórica anualizada."""
    log_returns = np.log(prices / prices.shift(1)).dropna()
    daily_vol = log_returns.std()
    annual_vol = daily_vol * np.sqrt(252)
    return annual_vol


# =============================================================================
# IV RANK - VOLATILITY CONE
# =============================================================================

def build_volatility_cone(prices: pd.Series, windows: list = None) -> dict:
    """
    Constrói o Volatility Cone calculando HV para múltiplas janelas.
    
    Args:
        prices: Série de preços (idealmente 252+ dias)
        windows: Lista de janelas em dias [10, 20, 30, 60, 90, 252]
    
    Returns:
        dict com estatísticas do cone para cada janela
    """
    if windows is None:
        windows = [10, 20, 30, 60, 90, 180, 252]
    
    log_returns = np.log(prices / prices.shift(1)).dropna()
    
    cone = {}
    all_hvs = []
    
    for window in windows:
        if len(log_returns) < window:
            continue
        
        # Calcula HV rolling para esta janela
        rolling_hv = log_returns.rolling(window).std() * np.sqrt(252)
        rolling_hv = rolling_hv.dropna()
        
        if len(rolling_hv) == 0:
            continue
        
        hvs = rolling_hv.values
        all_hvs.extend(hvs)
        
        cone[window] = {
            'min': float(np.min(hvs)),
            'p25': float(np.percentile(hvs, 25)),
            'median': float(np.median(hvs)),
            'p75': float(np.percentile(hvs, 75)),
            'max': float(np.max(hvs)),
            'current': float(hvs[-1]) if len(hvs) > 0 else 0
        }
    
    # Estatísticas globais do cone
    if all_hvs:
        cone['global'] = {
            'min': float(np.min(all_hvs)),
            'max': float(np.max(all_hvs)),
            'median': float(np.median(all_hvs))
        }
    
    return cone


def calculate_iv_rank(current_iv: float, prices: pd.Series, lookback: int = 252) -> dict:
    """
    Calcula IV Rank comparando IV atual com o Volatility Cone histórico.
    
    Args:
        current_iv: Volatilidade implícita atual (decimal, ex: 0.45 para 45%)
        prices: Série de preços para calcular HV histórica
        lookback: Período de lookback em dias
    
    Returns:
        dict com iv_rank, interpretação e dados do cone
    """
    cone = build_volatility_cone(prices)
    
    if 'global' not in cone:
        return {
            'iv_rank': 50,
            'interpretation': 'NEUTRO',
            'color': '#636EFA',
            'cone': cone,
            'hv_min': 0,
            'hv_max': 0
        }
    
    hv_min = cone['global']['min']
    hv_max = cone['global']['max']
    
    # Evita divisão por zero
    if hv_max == hv_min:
        iv_rank = 50
    else:
        iv_rank = (current_iv - hv_min) / (hv_max - hv_min) * 100
        iv_rank = max(0, min(100, iv_rank))
    
    # Interpretação
    if iv_rank >= 80:
        interpretation = "MUITO ALTA"
        color = "#39E58C"  # Verde - excelente para venda
        sell_signal = "Excelente"
    elif iv_rank >= 60:
        interpretation = "ALTA"
        color = "#00D4FF"  # Azul claro - bom para venda
        sell_signal = "Bom"
    elif iv_rank >= 40:
        interpretation = "NORMAL"
        color = "#636EFA"  # Azul - neutro
        sell_signal = "Neutro"
    elif iv_rank >= 20:
        interpretation = "BAIXA"
        color = "#FFB302"  # Amarelo - cautela
        sell_signal = "Cautela"
    else:
        interpretation = "MUITO BAIXA"
        color = "#FF4B4B"  # Vermelho - evitar
        sell_signal = "Evitar"
    
    return {
        'iv_rank': round(iv_rank, 1),
        'interpretation': interpretation,
        'color': color,
        'sell_signal': sell_signal,
        'cone': cone,
        'hv_min': round(hv_min * 100, 1),
        'hv_max': round(hv_max * 100, 1),
        'current_iv': round(current_iv * 100, 1)
    }


def calculate_iv_percentile(current_iv: float, prices: pd.Series, lookback: int = 252) -> float:
    """
    Calcula IV Percentile: % de dias que IV foi MENOR que a atual.
    
    Alternativa ao IV Rank, mais robusta a outliers.
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    
    # Calcula HV diária (usando janela de 20 dias como proxy)
    rolling_hv = log_returns.rolling(20).std() * np.sqrt(252)
    rolling_hv = rolling_hv.dropna().tail(lookback)
    
    if len(rolling_hv) == 0:
        return 50.0
    
    days_below = (rolling_hv < current_iv).sum()
    percentile = (days_below / len(rolling_hv)) * 100
    
    return round(percentile, 1)


# =============================================================================
# FRACTIONAL BROWNIAN MOTION MONTE CARLO
# =============================================================================

def generate_fbm_paths(S0: float, mu: float, sigma: float, H: float, 
                       T: float, n_steps: int, n_paths: int) -> np.ndarray:
    """
    Gera paths de preços usando Movimento Browniano Fracionário (fBm).
    
    Usa decomposição de Cholesky para gerar incrementos correlacionados
    baseados no expoente de Hurst.
    """
    dt = T / n_steps
    
    # Matriz de covariância para fBm
    t = np.arange(1, n_steps + 1) * dt
    
    cov = np.zeros((n_steps, n_steps))
    for i in range(n_steps):
        for j in range(n_steps):
            ti, tj = t[i], t[j]
            cov[i, j] = 0.5 * (ti**(2*H) + tj**(2*H) - abs(ti - tj)**(2*H))
    
    # Regularização para estabilidade numérica
    cov += np.eye(n_steps) * 1e-10
    
    # Decomposição de Cholesky
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        # Fallback para BM padrão se Cholesky falhar
        L = np.eye(n_steps) * np.sqrt(dt)
    
    # Gera valores fBm
    Z = np.random.standard_normal((n_paths, n_steps))
    fBm_values = Z @ L.T
    
    # Calcula incrementos
    fBm_increments = np.zeros_like(fBm_values)
    fBm_increments[:, 0] = fBm_values[:, 0]
    fBm_increments[:, 1:] = np.diff(fBm_values, axis=1)
    
    # Gera paths de preços usando fBm geométrico
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    
    for i in range(n_steps):
        paths[:, i+1] = paths[:, i] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * fBm_increments[:, i]
        )
    
    return paths


def run_monte_carlo_fbm(S0: float, K: float, r: float, sigma: float, 
                        H: float, T: float, n_paths: int = 3000) -> dict:
    """
    Executa simulação Monte Carlo usando fBm.
    
    Returns:
        dict com resultados da simulação
    """
    n_days = max(int(T * 365), 1)
    
    # Gera paths
    paths = generate_fbm_paths(S0, r, sigma, H, T, n_days, n_paths)
    
    # Preços finais
    final_prices = paths[:, -1]
    
    # Probabilidade de exercício (preço final < strike)
    prob_exercise = np.mean(final_prices < K)
    
    # Risco de Ruína: preço toca 10% abaixo do strike em algum momento
    ruin_level = K * 0.90
    min_prices = np.min(paths, axis=1)
    prob_ruin = np.mean(min_prices < ruin_level)
    
    # Estatísticas adicionais
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
# TREND FILTERS
# =============================================================================

def check_trend_filters(prices: pd.Series) -> dict:
    """
    Verifica 3 filtros de tendência para decisão de trading.
    
    Returns:
        dict com resultados dos filtros e valores
    """
    current_price = prices.iloc[-1]
    
    # Filtro A: Preço > SMA 21
    sma_21 = prices.tail(21).mean()
    filter_a = current_price > sma_21
    
    # Filtro B: Momentum 30 dias > 0
    if len(prices) >= 30:
        momentum_30 = (current_price / prices.iloc[-30] - 1) * 100
    else:
        momentum_30 = (current_price / prices.iloc[0] - 1) * 100
    filter_b = momentum_30 > 0
    
    # Filtro C: Slope da Regressão Linear 30 dias > 0
    prices_30 = prices.tail(30)
    x = np.arange(len(prices_30))
    y = prices_30.values
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    filter_c = slope > 0
    
    return {
        'filter_a': filter_a,
        'filter_b': filter_b,
        'filter_c': filter_c,
        'sma_21': sma_21,
        'momentum_30': momentum_30,
        'slope': slope,
        'r_squared': r_value ** 2,
        'all_bullish': filter_a and filter_b and filter_c
    }


# =============================================================================
# TRADING RECOMMENDATION
# =============================================================================

def get_recommendation(hurst: float, filters: dict, spot: float) -> tuple:
    """
    Gera recomendação de trading baseada em Hurst e filtros de tendência.
    
    Returns:
        tuple: (classificação, texto_recomendação, nível_risco, cor)
    """
    sma_21 = filters['sma_21']
    
    # Cenário PERSISTENTE (H > 0.55)
    if hurst > 0.55:
        if filters['all_bullish']:
            return (
                "VENDA FORTE",
                "Tendência de alta confirmada e persistente. Baixo risco de exercício.",
                "LOW",
                "#39E58C"
            )
        else:
            return (
                "RISCO ALTO",
                "Mercado persistente em queda ou sem direção clara. Perigo de movimento forte contra a Put.",
                "HIGH",
                "#FF4B4B"
            )
    
    # Cenário REVERSÃO (H < 0.45)
    elif hurst < 0.45:
        if spot < sma_21:
            return (
                "OPORTUNIDADE",
                "Ativo esticado para baixo com característica de reversão. Boa probabilidade de retorno.",
                "LOW",
                "#00D4FF"
            )
        else:
            return (
                "CAUTELA",
                "Ativo em exaustão de topo com tendência de voltar para a média.",
                "MEDIUM",
                "#FFB302"
            )
    
    # Cenário RUÍDO (0.45 <= H <= 0.55)
    else:
        return (
            "NEUTRO",
            "Siga estritamente as probabilidades do Delta de Black-Scholes tradicional.",
            "MEDIUM",
            "#636EFA"
        )
