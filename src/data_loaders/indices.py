
import pandas as pd
import requests
import json
import base64
import yfinance as yf
import streamlit as st

def parse_pt_br_float(s):
    try:
        if isinstance(s, (int, float)):
            return float(s)
        if isinstance(s, str):
            # Remove pontos de milhar e troca vírgula por ponto
            return float(s.replace('.', '').replace(',', '.'))
        return 0.0
    except:
        return 0.0

@st.cache_data(ttl=3600*25)
def fetch_index_composition(index_code):
    """
    Fetches the current composition of a B3 index using their internal API.
    index_code: str (e.g., 'ICON', 'IEEX', 'IMOB')
    """
    url_template = "https://sistemaswebb3-listados.b3.com.br/indexProxy/indexCall/GetPortfolioDay/{}"
    
    # Payload structure expected by B3
    payload_dict = {"index": index_code, "language": "pt-br"}
    payload_json = json.dumps(payload_dict)
    payload_b64 = base64.b64encode(payload_json.encode('utf-8')).decode('utf-8')
    
    url = url_template.format(payload_b64)
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # 'results' key contains the list of assets
        if 'results' not in data:
            return pd.DataFrame()
            
        assets = data['results']
        df = pd.DataFrame(assets)
        
        if df.empty:
             return pd.DataFrame()

        # Rename and clean
        df = df[['cod', 'theoricalQty']].copy()
        df.columns = ['Ticker', 'Qty']
        
        # Clean Ticker (add .SA)
        df['Ticker'] = df['Ticker'].astype(str).str.strip() + ".SA"
        
        # Clean Qty
        df['Qty'] = df['Qty'].apply(parse_pt_br_float)
        
        return df
        
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=3600*25)
def download_prices_sector(tickers, start_date):
    if not tickers:
        return pd.DataFrame()
    return yf.download(tickers, start=start_date, progress=False, threads=True)
