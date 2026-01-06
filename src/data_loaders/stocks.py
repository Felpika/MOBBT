
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

def obter_market_cap_individual(ticker):
    """
    Busca simples e direta no Yahoo Finance.
    """
    if pd.isna(ticker) or ticker == "SEM_TICKER":
        return ticker, np.nan
    
    ticker_clean = str(ticker).strip().upper()
    symbol = f"{ticker_clean}.SA" if not ticker_clean.endswith(".SA") else ticker_clean
        
    try:
        stock = yf.Ticker(symbol)
        mcap = None
        try:
            mcap = stock.fast_info.market_cap
        except:
            pass
            
        if pd.isna(mcap) or mcap is None or mcap == 0:
            try:
                mcap = stock.info.get('marketCap')
            except:
                pass

        return ticker, mcap
    except Exception:
        return ticker, np.nan

@st.cache_data(ttl=3600*4) 
def buscar_market_caps_otimizado(df_lookup, force_refresh=False):
    """
    Busca Market Caps em paralelo.
    """
    tickers = df_lookup['Codigo_Negociacao'].dropna().unique().tolist()
    tickers = [t for t in tickers if t != "SEM_TICKER"]
    
    resultados = {}
    
    if tickers:
        progresso = st.progress(0, text="Baixando valores de mercado...")
        total = len(tickers)
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_ticker = {
                executor.submit(obter_market_cap_individual, t): t 
                for t in tickers
            }
            
            for i, future in enumerate(as_completed(future_to_ticker)):
                ticker, cap = future.result()
                resultados[ticker] = cap
                progresso.progress((i + 1) / total, text=f"Processando: {ticker}")
                
        progresso.empty() 
    
    df_caps = pd.DataFrame(list(resultados.items()), columns=['Codigo_Negociacao', 'MarketCap'])
    return pd.merge(df_lookup, df_caps, on='Codigo_Negociacao', how='left')
