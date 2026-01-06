
import requests
import pandas as pd
import zipfile
import io
import streamlit as st
from datetime import date, timedelta
import json
import base64

@st.cache_data(ttl=300)  # Cache de 5 minutos
def fetch_option_price_b3(option_ticker, trade_date=None):
    """
    Busca o último preço negociado de uma opção na B3.
    
    Args:
        option_ticker: código da opção (ex: BOVAN159)
        trade_date: data no formato YYYY-MM-DD (default: último dia útil)
    
    Returns:
        dict com last_price, avg_price, volume, trades ou None se erro
    """
    
    if trade_date is None:
        # Calcula último dia útil
        today = date.today()
        if today.weekday() == 0:  # Segunda
            trade_date = (today - timedelta(days=3)).strftime("%Y-%m-%d")
        elif today.weekday() == 6:  # Domingo
            trade_date = (today - timedelta(days=2)).strftime("%Y-%m-%d")
        else:
            trade_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    
    url = f"https://arquivos.b3.com.br/rapinegocios/tickercsv/{option_ticker}/{trade_date}"
    
    try:
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            return None
        
        # Extrai ZIP
        z = zipfile.ZipFile(io.BytesIO(response.content))
        if not z.namelist():
            return None
        
        # Lê o arquivo CSV
        with z.open(z.namelist()[0]) as f:
            content = f.read().decode('latin-1')
            df = pd.read_csv(io.StringIO(content), sep=';', decimal=',')
        
        if df.empty or 'PrecoNegocio' not in df.columns:
            return None
        
        # Calcula estatísticas
        return {
            'last_price': df['PrecoNegocio'].iloc[-1],
            'avg_price': df['PrecoNegocio'].mean(),
            'min_price': df['PrecoNegocio'].min(),
            'max_price': df['PrecoNegocio'].max(),
            'volume': df['QuantidadeNegociada'].sum(),
            'trades': len(df),
            'date': trade_date
        }
    except Exception as e:
        return None
