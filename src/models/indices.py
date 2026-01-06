
import pandas as pd
import streamlit as st

def calculate_sector_deviation(compositions, prices, sector):
    """
    Calcula o desvio do índice setorial em relação à sua média móvel de 50 dias.
    """
    comp_df = compositions[sector]
    valid_tickers = [t for t in comp_df['Ticker'] if t in prices.columns]
    
    if not valid_tickers:
        return None
    
    sector_slice = prices[valid_tickers]
    
    # Filter bad tickers (>80% NaN)
    total_rows = len(sector_slice)
    valid_counts = sector_slice.count()
    bad_tickers = valid_counts[valid_counts < 0.8 * total_rows].index.tolist()
    valid_tickers = [t for t in valid_tickers if t not in bad_tickers]
    
    if not valid_tickers:
         return None
         
    sector_slice = prices[valid_tickers]
    sector_slice = sector_slice.dropna(axis=1, how='all')
    valid_tickers = sector_slice.columns.tolist()

    if not valid_tickers:
         return None

    # Forward fill
    sector_prices = sector_slice.ffill()
    
    # Get weights
    comp_df_sector = comp_df.set_index('Ticker')
    valid_weights_keys = [t for t in valid_tickers if t in comp_df_sector.index]
    weights = comp_df_sector.loc[valid_weights_keys, 'Qty']
    
    sector_prices = sector_prices[valid_weights_keys]
    
    # Calculate Index Value
    try:
         sector_val = sector_prices.dot(weights)
    except Exception:
         return None
    
    # Calculate MA50
    ma50 = sector_val.rolling(window=50).mean()
    
    # Calculate Deviation
    deviation = ((sector_val - ma50) / ma50) * 100
    return deviation
