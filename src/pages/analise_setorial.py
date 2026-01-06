
import streamlit as st
import pandas as pd
from src.data_loaders.indices import fetch_index_composition, download_prices_sector
from src.models.indices import calculate_sector_deviation
from src.components.charts import plot_sector_indices_chart
from src.config.py import COLORS # Assuming this import works if config is standard module

def render():
    st.title("Análise Setorial (Smart Money Flow)")
    st.markdown("---")
    
    # 1. Configuration 
    start_date = "2022-01-01" 
    
    index_meta = {
        'IMOB': {'color': '#39E58C', 'name': 'Imobiliário (IMOB)'},       
        'IFNC': {'color': '#00D4FF', 'name': 'Financeiro (IFNC)'},        
        'ICON': {'color': '#F0F6FC', 'name': 'Consumo (ICON)'},           
        'UTIL': {'color': '#FF4B4B', 'name': 'Utilidade Pública (UTIL)'}, 
        'IEEX': {'color': '#FFB302', 'name': 'Energia Elétrica (IEEX)'},  
        'IMAT': {'color': '#AB47BC', 'name': 'Materiais Básicos (IMAT)'}, 
        'INDX': {'color': '#5C6BC0', 'name': 'Indústria (INDX)'}          
    }
    
    compositions = {}
    all_tickers = set()
    
    # 2. Fetch Compositions
    progress_col = st.empty()
    bar = st.progress(0)
    
    total_indices = len(index_meta)
    
    for i, sector_code in enumerate(index_meta.keys()):
        progress_col.text(f"Baixando composição: {sector_code}...")
        df = fetch_index_composition(sector_code)
        if not df.empty:
            compositions[sector_code] = df
            all_tickers.update(df['Ticker'].tolist())
        bar.progress((i + 1) / (total_indices * 2))
            
    if not all_tickers:
        bar.empty()
        progress_col.empty()
        st.error("Não foi possível encontrar tickers para nenhum setor.")
        return

    # 3. Download Data
    progress_col.text(f"Baixando preços de {len(all_tickers)} ativos...")
    bar.progress(0.6)
    
    try:
        data = download_prices_sector(list(all_tickers), start_date)
        
        # Treatment for yfinance output
        prices = pd.DataFrame(index=data.index)
        if isinstance(data.columns, pd.MultiIndex):
            # Try parsing multi-level columns
            for ticker in all_tickers:
                if ('Adj Close', ticker) in data.columns:
                    prices[ticker] = data[('Adj Close', ticker)]
                elif ('Close', ticker) in data.columns:
                    prices[ticker] = data[('Close', ticker)]
        else:
             if 'Adj Close' in data.columns: prices = data['Adj Close']
             elif 'Close' in data.columns: prices = data['Close']
             else: prices = data

    except Exception as e:
        bar.empty()
        progress_col.empty()
        st.error(f"Falha ao baixar dados de preços: {e}")
        return

    # 4. Calculate Indices
    progress_col.text("Calculando desvios dos índices setoriais...")
    bar.progress(0.8)
    
    results = pd.DataFrame()
    
    for sector in compositions.keys():
        deviation = calculate_sector_deviation(compositions, prices, sector)
        if deviation is not None:
             results[sector] = deviation

    bar.empty()
    progress_col.empty()

    if results.empty:
        st.warning("Não foi possível calcular índices setoriais.")
        return

    # 5. Plotting
    fig = plot_sector_indices_chart(results, index_meta)
    st.plotly_chart(fig, use_container_width=True)
