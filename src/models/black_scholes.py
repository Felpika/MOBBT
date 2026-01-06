
import numpy as np
from scipy.stats import norm

def black_scholes_put(S, K, T, r, sigma):
    """
    Calcula preço teórico de PUT usando Black-Scholes.
    
    Args:
        S: Preço do ativo
        K: Strike
        T: Tempo até vencimento em anos
        r: Taxa livre de risco anual (decimal)
        sigma: Volatilidade (decimal)
    """
    if T <= 0:
        return max(K - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

def implied_volatility(market_price, S, K, T, r, max_iter=100, tol=1e-5):
    """
    Calcula volatilidade implícita usando Newton-Raphson.
    """
    sigma = 0.3  # Chute inicial
    for i in range(max_iter):
        price = black_scholes_put(S, K, T, r, sigma)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        
        if vega < 1e-10:
            break
            
        diff = market_price - price
        if abs(diff) < tol:
            return sigma
        sigma = sigma + diff / vega
        sigma = max(0.01, min(sigma, 3.0))  # Limita entre 1% e 300%
    return sigma

def calculate_greeks(S, K, T, r, sigma, option_type='put'):
    """
    Calcula as Gregas (Delta, Gamma, Vega, Theta) para uma opção.
    Retorna um dicionário com os valores.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Common Greeks
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Scaled for 1% change
    
    if option_type == 'put':
        delta = norm.cdf(d1) - 1
        theta_annual = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                        + r * K * np.exp(-r * T) * norm.cdf(-d2))
        theta_daily = theta_annual / 365
    else: # call
        delta = norm.cdf(d1)
        theta_annual = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                        - r * K * np.exp(-r * T) * norm.cdf(d2))
        theta_daily = theta_annual / 365
        
    prob_exercise = abs(delta) * 100
    
    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta_daily': theta_daily,
        'param_iv': sigma,
        'prob_exercise': prob_exercise
    }
