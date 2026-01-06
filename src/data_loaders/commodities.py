
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import timedelta
import numpy as np

@st.cache_data(ttl=3600*4)
def carregar_dados_commodities():
    commodities_map = {
        'Petróleo Brent': 'BZ=F', 
        'Cacau': 'CC=F', 
        'Petróleo WTI': 'CL=F', 
        'Algodão': 'CT=F', 
        'Ouro': 'GC=F', 
        'Cobre': 'HG=F', 
        'Óleo de Aquecimento': 'HO=F', 
        'Café': 'KC=F', 
        'Trigo (KC HRW)': 'KE=F', 
        'Madeira': 'LBS=F', 
        'Gado Bovino': 'LE=F', 
        'Gás Natural': 'NG=F', 
        'Suco de Laranja': 'OJ=F', 
        'Paládio': 'PA=F', 
        'Platina': 'PL=F', 
        'Gasolina RBOB': 'RB=F', 
        'Açúcar': 'SB=F', 
        'Prata': 'SI=F', 
        'Milho': 'ZC=F', 
        'Óleo de Soja': 'ZL=F', 
        'Aveia': 'ZO=F', 
        'Arroz': 'ZR=F', 
        'Soja': 'ZS=F'
    }
    dados_commodities_raw = {}
    with st.spinner("Baixando dados históricos de commodities... (cache de 4h)"):
        for nome, ticker in commodities_map.items():
            try:
                dado = yf.download(ticker, period='max', auto_adjust=True, progress=False)
                if not dado.empty: dados_commodities_raw[nome] = dado['Close']
            except Exception: pass
            
    categorized_commodities = {
        'Energia': ['Petróleo Brent', 'Petróleo WTI', 'Óleo de Aquecimento', 'Gás Natural', 'Gasolina RBOB'], 
        'Metais Preciosos': ['Ouro', 'Paládio', 'Platina', 'Prata'], 
        'Metais Industriais': ['Cobre'], 
        'Agricultura': ['Cacau', 'Algodão', 'Café', 'Trigo (KC HRW)', 'Madeira', 'Gado Bovino', 'Suco de Laranja', 'Açúcar', 'Milho', 'Óleo de Soja', 'Aveia', 'Arroz', 'Soja']
    }
    dados_por_categoria = {}
    for categoria, nomes in categorized_commodities.items():
        series_da_categoria = {nome: dados_commodities_raw[nome] for nome in nomes if nome in dados_commodities_raw}
        if series_da_categoria:
            df_cat = pd.concat(series_da_categoria, axis=1)
            df_cat.columns = series_da_categoria.keys()
            dados_por_categoria[categoria] = df_cat
    return dados_por_categoria

def calcular_variacao_commodities(dados_por_categoria):
    all_series = [s for df in dados_por_categoria.values() for s in [df[col].dropna() for col in df.columns]]
    if not all_series: return pd.DataFrame()
    df_full = pd.concat(all_series, axis=1)
    df_full.sort_index(inplace=True)
    if df_full.empty: return pd.DataFrame()
    latest_date = df_full.index.max()
    latest_prices = df_full.loc[latest_date]
    periods = {'1 Dia': 1, '1 Semana': 7, '1 Mês': 30, '3 Meses': 91, '6 Meses': 182, '1 Ano': 365}
    results = []
    for name in df_full.columns:
        res = {'Commodity': name, 'Preço Atual': latest_prices[name]}
        series = df_full[name].dropna()
        for label, days in periods.items():
            past_date = latest_date - timedelta(days=days)
            past_price = series.asof(past_date)
            res[f'Variação {label}'] = ((latest_prices[name] - past_price) / past_price) if pd.notna(past_price) and past_price > 0 else np.nan
        results.append(res)
    return pd.DataFrame(results).set_index('Commodity')
