
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from src.data_loaders.indices import fetch_index_composition, download_prices_sector
from src.components.charts import plot_sector_indices_chart
from datetime import datetime, timedelta

def calculate_sector_deviation(compositions, prices, sector):
    """
    Calcula o desvio do índice setorial em relação à sua média móvel de 50 dias.
    """
    comp_df = compositions[sector]
    valid_tickers = [t for t in comp_df['Ticker'] if t in prices.columns]
    
    if not valid_tickers: return None
    
    sector_slice = prices[valid_tickers]
    total_rows = len(sector_slice)
    valid_counts = sector_slice.count()
    bad_tickers = valid_counts[valid_counts < 0.8 * total_rows].index.tolist()
    valid_tickers = [t for t in valid_tickers if t not in bad_tickers]
    
    if not valid_tickers: return None
         
    sector_slice = prices[valid_tickers]
    sector_slice = sector_slice.dropna(axis=1, how='all')
    valid_tickers = sector_slice.columns.tolist()

    if not valid_tickers: return None

    sector_prices = sector_slice.ffill()
    comp_df_sector = comp_df.set_index('Ticker')
    valid_weights_keys = [t for t in valid_tickers if t in comp_df_sector.index]
    weights = comp_df_sector.loc[valid_weights_keys, 'Qty']
    sector_prices = sector_prices[valid_weights_keys]
    
    try:
         sector_val = sector_prices.dot(weights)
    except Exception:
         return None
    
    ma50 = sector_val.rolling(window=50).mean()
    deviation = ((sector_val - ma50) / ma50) * 100
    return deviation

def get_sector_indices_chart():
    # 1. Configuration 
    start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
    
    INDICES = {
        'IMAT': {'name': 'Materiais Básicos', 'color': '#FF1744'}, # Red
        'ICON': {'name': 'Consumo', 'color': '#FF9100'}, # Orange
        'IEEX': {'name': 'Energia Elétrica', 'color': '#2979FF'}, # Blue
        'IMOB': {'name': 'Imobiliário', 'color': '#D500F9'}, # Purple
        'IFNC': {'name': 'Financeiro', 'color': '#00E676'}, # Green
        'UTIL': {'name': 'Utilidade Pública', 'color': '#00E5FF'}, # Cyan
        'INDX': {'name': 'Industrial', 'color': '#FFEA00'} # Yellow
    }

    # 2. Fetch Compositions
    compositions = {}
    all_tickers = []
    
    for idx_code in INDICES.keys():
        comp_df = fetch_index_composition(idx_code)
        if not comp_df.empty:
            compositions[idx_code] = comp_df
            all_tickers.extend(comp_df['Ticker'].tolist())
    
    all_tickers = list(set(all_tickers))
    
    # 3. Download Prices
    if not all_tickers:
        return go.Figure().update_layout(title_text="Sem tickers para índices setoriais")

    data = download_prices_sector(all_tickers, start_date)
    
    if data.empty:
        return go.Figure().update_layout(title_text="Falha ao baixar preços dos índices")
        
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    
    # 4. Calculate Indices & Deviations
    results = pd.DataFrame(index=prices.index)
    
    for sector in INDICES.keys():
        if sector in compositions:
            dev = calculate_sector_deviation(compositions, prices, sector)
            if dev is not None:
                results[sector] = dev
    
    # 5. Plot
    return plot_sector_indices_chart(results, INDICES)
