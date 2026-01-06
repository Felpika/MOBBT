
import pandas as pd
from fredapi import Fred
import streamlit as st

@st.cache_data(ttl=3600*4)
def carregar_dados_fred(api_key, tickers_dict):
    """
    Carrega dados do FRED dado um dicionário de tickers.
    """
    fred = Fred(api_key=api_key)
    lista_series = []
    st.info("Carregando dados do FRED... (Cache de 4h)")
    for ticker in tickers_dict.keys():
        try:
            serie = fred.get_series(ticker)
            serie.name = ticker
            lista_series.append(serie)
        except Exception as e: 
            st.warning(f"Não foi possível carregar o ticker '{ticker}' do FRED: {e}")
            
    if not lista_series: return pd.DataFrame()
    return pd.concat(lista_series, axis=1).ffill()
