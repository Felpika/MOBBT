import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from bcb import sgs
from datetime import datetime, timedelta
import os
from fredapi import Fred
import requests
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIGURAÇÃO GERAL DA PÁGINA ---
st.set_page_config(layout="wide", page_title="MOBBT")

# --- BLOCO 1: LÓGICA DO DASHBOARD DO TESOURO DIRETO ---
@st.cache_data(ttl=3600*4)
def obter_dados_tesouro():
    url = 'https://www.tesourotransparente.gov.br/ckan/dataset/df56aa42-484a-4a59-8184-7676580c81e3/resource/796d2059-14e9-44e3-80c9-2d9e30b405c1/download/precotaxatesourodireto.csv'
    st.info("Carregando dados do Tesouro Direto... (Cache de 4h)")
    try:
        df = pd.read_csv(url, sep=';', decimal=',')
        df['Data Vencimento'] = pd.to_datetime(df['Data Vencimento'], format='%d/%m/%Y')
        df['Data Base'] = pd.to_datetime(df['Data Base'], format='%d/%m/%Y')
        df['Tipo Titulo'] = df['Tipo Titulo'].astype('category')
        return df
    except Exception as e:
        st.error(f"Erro ao baixar dados do Tesouro: {e}")
        return pd.DataFrame()

@st.cache_data
def calcular_juro_10a_br(df_tesouro):
    df_ntnb = df_tesouro['Tipo Titulo'] == 'Tesouro IPCA+ com Juros Semestrais'
    df_ntnb = df_tesouro.loc df_ntnb.copy()
    if df_ntnb.empty:
        return pd.Series(dtype=float)
    resultados = {}
    for data_base in df_ntnb['Data Base'].unique():
        df_dia = df_ntnb['Data Base'] == data_base
        df_dia = df_ntnb.loc df_dia.copy()
        vencimentos_do_dia = df_dia['Data Vencimento'].unique()
        if len(vencimentos_do_dia) > 0:
            target_10y = pd.to_datetime(data_base) + pd.DateOffset(years=10)
            venc_10y = min(vencimentos_do_dia, key=lambda d: abs(d - target_10y))
            taxa = df_dia['Data Vencimento'] == venc_10y
            taxa = df_dia.loc taxa['Taxa Compra Manha'].iloc [0]
            resultados.update({data_base: taxa})
    return pd.Series(resultados).sort_index()

def gerar_grafico_historico_tesouro(df, tipo, vencimento, metrica='Taxa Compra Manha'):
    tipo_mask = df['Tipo Titulo'] == tipo
    venc_mask = df['Data Vencimento'] == vencimento
    df_filtrado = df.loc tipo_mask & venc_mask.copy()
    df_filtrado = df_filtrado.sort_values('Data Base')
    titulo = f'Histórico da Taxa de Compra: {tipo} (Venc. {vencimento.strftime("%d/%m/%Y")})' if metrica == 'Taxa Compra Manha' else f'Histórico do Preço Unitário (PU): {tipo} (Venc. {vencimento.strftime("%d/%m/%Y")})'
    eixo_y = "Taxa de Compra (% a.a.)" if metrica == 'Taxa Compra Manha' else "Preço Unitário (R$)"
    fig = px.line(df_filtrado, x='Data Base', y=metrica, title=titulo, template='plotly_dark')
    fig.update_layout(title_x=0, yaxis_title=eixo_y, xaxis_title="Data")
    return fig

@st.cache_data
def calcular_inflacao_implicita(df):
    data_max = df['Data Base'].max()
    df_recente = df['Data Base'] == data_max
    df_recente = df.loc df_recente.copy()
    tipos_ipca = ['Tesouro IPCA+ com Juros Semestrais', 'Tesouro IPCA+']
    ipca_mask = df_recente['Tipo Titulo'].isin(tipos_ipca)
    df_ipca_raw = df_recente.loc ipca_mask.copy()
    prefixado_mask = df_recente['Tipo Titulo'] == 'Tesouro Prefixado'
    df_prefixados = df_recente.loc prefixado_mask.copy()
    df_prefixados = df_prefixados.set_index('Data Vencimento')
    df_ipca = df_ipca_raw.sort_values('Tipo Titulo', ascending=False).drop_duplicates('Data Vencimento').set_index('Data Vencimento')
    if df_prefixados.empty or df_ipca.empty:
        return pd.DataFrame()
    inflacao_implicita = []
    for venc_prefixado, row_prefixado in df_prefixados.iterrows():
        venc_ipca_proximo = min(df_ipca.index, key=lambda d: abs(d - venc_prefixado))
        if abs((venc_ipca_proximo - venc_prefixado).days) < 550:
            taxa_prefixada = row_prefixado['Taxa Compra Manha']
            taxa_ipca = df_ipca.loc venc_ipca_proximo['Taxa Compra Manha']
            breakeven = (((1 + taxa_prefixada / 100) / (1 + taxa_ipca / 100)) - 1) * 100
            inflacao_implicita.append({'Vencimento do Prefixo': venc_prefixado, 'Inflação Implícita (% a.a.)': breakeven})
    if not inflacao_implicita:
        return pd.DataFrame()
    return pd.DataFrame(inflacao_implicita).sort_values('Vencimento do Prefixo').set_index('Vencimento do Prefixo')

@st.cache_data
def gerar_grafico_spread_juros(df):
    ntnf_mask = df['Tipo Titulo'] == 'Tesouro Prefixado com Juros Semestrais'
    df_ntnf = df.loc ntnf_mask.copy()
    if df_ntnf.empty:
        return go.Figure().update_layout(title_text="Não há dados de Tesouro Prefixado com Juros Semestrais.")
    data_recente = df_ntnf['Data Base'].max()
    titulos_hoje_mask = df_ntnf['Data Base'] == data_recente
    titulos_disponiveis_hoje = df_ntnf.loc titulos_hoje_mask
    vencimentos_atuais = sorted(titulos_disponiveis_hoje['Data Vencimento'].unique())
    if len(vencimentos_atuais) < 2:
        return go.Figure().update_layout(title_text="Menos de duas NTN-Fs disponíveis para calcular o spread.")
    target_2y = data_recente + pd.DateOffset(years=2)
    target_10y = data_recente + pd.DateOffset(years=10)
    venc_curto = min(vencimentos_atuais, key=lambda d: abs(d - target_2y))
    venc_longo = min(vencimentos_atuais, key=lambda d: abs(d - target_10y))
    if venc_curto == venc_longo:
        return go.Figure().update_layout(title_text="Não foi possível encontrar vértices de 2 e 10 anos distintos.")
    venc_curto_hist_mask = df_ntnf['Data Vencimento'] == venc_curto
    df_curto_hist = df_ntnf.loc venc_curto_hist_mask[['Data Base', 'Taxa Compra Manha']].set_index('Data Base')
    venc_longo_hist_mask = df_ntnf['Data Vencimento'] == venc_longo
    df_longo_hist = df_ntnf.loc venc_longo_hist_mask[['Data Base', 'Taxa Compra Manha']].set_index('Data Base')
    df_spread = pd.merge(df_curto_hist, df_longo_hist, on='Data Base', suffixes=('_curto', '_longo')).dropna()
    if df_spread.empty:
        return go.Figure().update_layout(title_text=f"Não há histórico comum entre as NTN-Fs.")
    df_spread['Spread'] = (df_spread['Taxa Compra Manha_longo'] - df_spread['Taxa Compra Manha_curto']) * 100
    fig = px.area(df_spread, y='Spread', title=f'Spread de Juros: NTN-F ~10 Anos ({pd.to_datetime(venc_longo).year}) vs ~2 Anos ({pd.to_datetime(venc_curto).year})', template='plotly_dark')
    fig.update_layout(title_x=0, yaxis_title="Diferença (Basis Points)", xaxis_title="Data", showlegend=False)
    return fig

def gerar_grafico_ettj_curto_prazo(df):
    prefixado_mask = df['Tipo Titulo'] == 'Tesouro Prefixado'
    df_prefixado = df.loc prefixado_mask.copy()
    if df_prefixado.empty:
        return go.Figure().update_layout(title_text="Não há dados para 'Tesouro Prefixado'.")
    datas_disponiveis = sorted(df_prefixado['Data Base'].unique())
    data_recente = datas_disponiveis[-1]
    targets = {f'Hoje ({data_recente.strftime("%d/%m/%Y")})': data_recente, '1 dia Atrás': data_recente - pd.DateOffset(days=1),
               '2 dias Atrás': data_recente - pd.DateOffset(days=2), '3 dias Atrás': data_recente - pd.DateOffset(days=3),
               '4 dias Atrás': data_recente - pd.DateOffset(days=4), '5 dias Atrás': data_recente - pd.DateOffset(days=5)}
    datas_para_plotar = {}
    for legenda_base, data_alvo in targets.items():
        datas_validas = [d for d in datas_disponiveis if d <= data_alvo]
        if datas_validas:
            data_real = max(datas_validas)
            if data_real not in datas_para_plotar.values():
                legenda_final = f'{" ".join(legenda_base.split(" ")[:2])} ({data_real.strftime("%d/%m/%Y")})' if 'Atrás' in legenda_base else legenda_base
                datas_para_plotar.update({legenda_final: data_real})
    fig = go.Figure()
    for legenda, data_base in datas_para_plotar.items():
        data_mask = df_prefixado['Data Base'] == data_base
        df_data = df_prefixado.loc data_mask.copy().sort_values('Data Vencimento')
        df_data['Dias Uteis'] = np.busday_count(df_data['Data Base'].values.astype('M8<0xC2><0xAD>[D]'), df_data['Data Vencimento'].values.astype('M8<0xC2><0xAD>[D]'))
        line_style = dict(dash='dash') if not legenda.startswith('Hoje') else {}
        fig.add_trace(go.Scatter(x=df_data['Dias Uteis'], y=df_data['Taxa Compra Manha'], mode='lines+markers', name=legenda, line=line_style))
    fig.update_layout(title_text='Curva de Juros (ETTJ) - Curto Prazo (últimos 5 dias)', title_x=0, xaxis_title='Dias Úteis até o Vencimento', yaxis_title='Taxa (% a.a.)', template='plotly_dark', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def gerar_grafico_ettj_longo_prazo(df):
    prefixado_mask = df['Tipo Titulo'] == 'Tesouro Prefixado'
    df_prefixado = df.loc prefixado_mask.copy()
    if df_prefixado.empty:
        return go.Figure().update_layout(title_text="Não há dados para 'Tesouro Prefixado'.")
    datas_disponiveis = sorted(df_prefixado['Data Base'].unique())
    data_recente = datas_disponiveis[-1]
    targets = {f'Hoje ({data_recente.strftime("%d/%m/%Y")})': data_recente, '1 Semana Atrás': data_recente - pd.DateOffset(weeks=1),
               '1 Mês Atrás': data_recente - pd.DateOffset(months=1), '3 Meses Atrás': data_recente - pd.DateOffset(months=3),
               '6 Meses Atrás': data_recente - pd.DateOffset(months=6), '1 Ano Atrás': data_recente - pd.DateOffset(years=1)}
    datas_para_plotar = {}
    for legenda_base, data_alvo in targets.items():
        datas_validas = [d for d in datas_disponiveis if d <= data_alvo]
        if datas_validas:
            data_real = max(datas_validas)
            if data_real not in datas_para_plotar.values():
                legenda_final = f'{" ".join(legenda_base.split(" ")[:2])} ({data_real.strftime("%d/%m/%Y")})' if not legenda_base.startswith('Hoje') else legenda_base
                datas_para_plotar.update({legenda_final: data_real})
    fig = go.Figure()
    for legenda, data_base in datas_para_plotar.items():
        data_mask = df_prefixado['Data Base'] == data_base
        df_data = df_prefixado.loc data_mask.copy().sort_values('Data Vencimento')
        df_data['Dias Uteis'] = np.busday_count(df_data['Data Base'].values.astype('M8<0xC2><0xAD>[D]'), df_data['Data Vencimento'].values.astype('M8<0xC2><0xAD>[D]'))
        line_style = dict(dash='dash') if not legenda.startswith('Hoje') else {}
        fig.add_trace(go.Scatter(x=df_data['Dias Uteis'], y=df_data['Taxa Compra Manha'], mode='lines+markers', name=legenda, line=line_style))
    fig.update_layout(title_text='Curva de Juros (ETTJ) - Longo Prazo (Comparativo Histórico)', title_x=0, xaxis_title='Dias Úteis até o Vencimento', yaxis_title='Taxa (% a.a.)', template='plotly_dark', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

# --- BLOCO 2: LÓGICA DO DASHBOARD DE INDICADORES ECONÔMICOS ---
@st.cache_data(ttl=3600*4)
def carregar_dados_bcb():
    SERIES_CONFIG = {'Spread Bancário': {'id': 20783}, 'Inadimplência': {'id': 21082}, 'Crédito/PIB': {'id': 20622},
                     'Juros Médio': {'id': 20714}, 'Confiança Consumidor': {'id': 4393}, 'IPCA': {'id': 16122},
                     'Atraso 15-90d Total': {'id': 21006}, 'Atraso 15-90d Agro': {'id': 21069},
                     'Inadimplência Crédito Rural': {'id': 21146}}
    lista_dfs_sucesso, config_sucesso = [], {}
    for name, config in SERIES_CONFIG.items():
        try:
            df_temp = sgs.get({name: config['id']}, start='2010-01-01')
            lista_dfs_sucesso.append(df_temp)
            config_sucesso.update({name: config})
        except Exception as e:
            st.warning(f"Não foi possível carregar o indicador '{name}': {e}")
    if not lista_dfs_sucesso:
        return pd.DataFrame(), {}
    df_full = pd.concat(lista_dfs_sucesso, axis=1)
    df_full.ffill(inplace=True)
    df_full.dropna(inplace=True)
    return df_full, config_sucesso

# --- BLOCO 3: LÓGICA DO DASHBOARD DE COMMODITIES ---
@st.cache_data(ttl=3600*4)
def carregar_dados_commodities():
    commodities_map = {'Petróleo Brent': 'BZ=F', 'Cacau': 'CC=F', 'Petróleo WTI': 'CL=F', 'Algodão': 'CT=F', 'Ouro': 'GC=F',
                       'Cobre': 'HG=F', 'Óleo de Aquecimento': 'HO=F', 'Café': 'KC=F', 'Trigo (KC HRW)': 'KE=F',
                       'Madeira': 'LBS=F', 'Gado Bovino': 'LE=F', 'Gás Natural': 'NG=F', 'Suco de Laranja': 'OJ=F',
                       'Paládio': 'PA=F', 'Platina': 'PL=F', 'Gasolina RBOB': 'RB=F', 'Açúcar': 'SB=F', 'Prata': 'SI=F',
                       'Milho': 'ZC=F', 'Óleo de Soja': 'ZL=F', 'Aveia': 'ZO=F', 'Arroz': 'ZR=F', 'Soja': 'ZS=F'}
    dados_commodities_raw = {}
    with st.spinner("Baixando dados históricos de commodities... (cache de 4h)"):
        for nome, ticker in commodities_map.items():
            try:
                dado = yf.download(ticker, period='max', auto_adjust=True, progress=False)
                if not dado.empty:
                    dados_commodities_raw.update({nome: dado['Close']})
            except Exception:
                pass
    categorized_commodities = {'Energia': ['Petróleo Brent', 'Petróleo WTI', 'Óleo de Aquecimento', 'Gás Natural', 'Gasolina RBOB'],
                               'Metais Preciosos': ['Ouro', 'Paládio', 'Platina', 'Prata'],
                               'Metais Industriais': ['Cobre'],
                               'Agricultura': ['Cacau', 'Algodão', 'Café', 'Trigo (KC HRW)', 'Madeira', 'Gado Bovino',
                                               'Suco de Laranja', 'Açúcar', 'Milho', 'Óleo de Soja', 'Aveia', 'Arroz',
                                               'Soja']}
    dados_por_categoria = {}
    for categoria, nomes in categorized_commodities.items():
        series_da_categoria = {nome: dados_commodities_raw.get(nome) for nome in nomes if nome in dados_commodities_raw}
        if series_da_categoria:
            df_cat = pd.concat(series_da_categoria, axis=1)
            df_cat.columns = series_da_categoria.keys()
            dados_por_categoria.update({categoria: df_cat})
    return dados_por_categoria

def calcular_variacao_commodities(dados_por_categoria):
    all_series = [s.dropna() for df in dados_por_categoria.values() for col in df.columns for s in [df.loc :, col]]
    if not all_series:
        return pd.DataFrame()
    df_full = pd.concat(all_series, axis=1)
    df_full.sort_index(inplace=True)
    if df_full.empty:
        return pd.DataFrame()
    latest_date = df_full.index.max()
    latest_prices = df_full.loc latest_date
    periods = {'1 Dia': 1, '1 Semana': 7, '1 Mês': 30, '3 Meses': 91, '6 Meses': 182, '1 Ano': 365}
    results = []
    for name in df_full.columns:
        res = {'Commodity': name, 'Preço Atual': latest_prices.get(name)}; series = df_full.get(name).dropna()
        for label, days in periods.items():
            past_date = latest_date - timedelta(days=days)
            past_price = series.asof(past_date)
            res.update({f'Variação {label}': ((latest_prices.get(name) - past_price) / past_price) if pd.notna(past_price) and past_price > 0 else np.nan})
        results.append(res)
    return pd.DataFrame(results).set_index('Commodity')

def colorir_negativo_positivo(val):
    if pd.isna(val) or val == 0:
        return ''
    return f"color: {'#4CAF50' if val > 0 else '#F44336'}"

def gerar_dashboard_commodities(dados_preco_por_categoria):
    all_commodity_names = [name for df in dados_preco_por_categoria.values() for name in df.columns]
    total_subplots = len(all_commodity_names)
    if total_subplots == 0:
        return go.Figure().update_layout(title_text="Nenhum dado de commodity disponível.")
    num_cols, num_rows = 4, int(np.ceil(total_subplots / 4))
    fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=all_commodity_names)
    idx = 0
    for df_cat in dados_preco_por_categoria.values():
        for commodity_name in df_cat.columns:
            row, col = (idx // num_cols) + 1, (idx % num_cols) + 1
            fig.add_trace(go.Scatter(x=df_cat.index, y=df_cat.get(commodity_name), mode='lines', name=commodity_name), row=row, col=col)
            idx += 1
    end_date = datetime.now()
    buttons = []
    periods = {'1M': 30, '3M': 91, '6M': 182, 'YTD': 'ytd', '1A': 365, '5A': 365 * 5, '10A': 3650, 'Máx': 'max'}
    for label, days in periods.items():
        if days == 'ytd':
            start_date = datetime(end_date.year, 1, 1)
        elif days == 'max':
            start_date = min([df.index.min() for df in dados_preco_por_categoria.values() if not df.empty])
        else:
            start_date = end_date - timedelta(days=days)
        update_args = {}
        for i in range(1, total_subplots + 1):
            update_args.update({f'xaxis{i if i > 1 else ""}.range': [start_date, end_date], f'yaxis{i if i > 1 else ""}.autorange': True})
        buttons.append(dict(method='relayout', label=label, args=[update_args]))
    active_button_index = list(periods.keys()).index('1A') if '1A' in list(periods.keys()) else 4
    fig.update_layout(title_text="Dashboard de Preços Históricos de Commodities", title_x=0, template="plotly_dark",
                      height=250 * num_rows, showlegend=False,
                      updatemenus=[dict(type="buttons", direction="right", showactive=True, x=1, xanchor="right", y=1.05,
                                        yanchor="bottom", buttons=buttons, active=active_button_index)])
    start_date_1y = end_date - timedelta(days=365)
    idx = 0
    for df_cat in dados_preco_por_categoria.values():
        for i, commodity_name in enumerate(df_cat.columns, start=idx):
            fig.layout.update({f'xaxis{i + 1 if i + 1 > 1 else ""}.range': [start_date_1y, end_date]})
            series = df_cat.get(commodity_name)
            filtered_series = series[(series.index >= start_date_1y) & (series.index <= end_date)].dropna()
            if not filtered_series.empty:
                min_y, max_y = filtered_series.min(), filtered_series.max()
                padding = (max_y - min_y) * 0.05
                fig.layout.update({f'yaxis{i + 1 if i + 1 > 1 else ""}.range': [min_y - padding, max_y + padding]})
            else:
                fig.layout.update({f'yaxis{i + 1 if i + 1 > 1 else ""}.autorange': True})
        idx += len(df_cat.columns)
    return fig

# --- BLOCO 4: LÓGICA DO DASHBOARD DE INDICADORES INTERNACIONAIS ---
@st.cache_data(ttl=3600*4)
def carregar_dados_fred(api_key, tickers_dict):
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
    if not lista_series:
        return pd.DataFrame()
    return pd.concat(lista_series, axis=1).ffill()

def gerar_grafico_fred(df, ticker, titulo):
    if ticker not in df.columns or df.get(ticker).isnull().all():
        return go.Figure().update_layout(title_text=f"Dados para {ticker} não encontrados.")
    fig = px.line(df, y=ticker, title=titulo, template='plotly_dark')
    if ticker == 'T10Y2Y':
        fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Inversão",
                      annotation_position="bottom right")
    end_date = df.index.max()
    buttons = []
    periods = {'6M': 182, '1A': 365, '2A': 730, '5A': 1825, '10A': 3650, 'Máx': 'max'}
    for label, days in periods.items():
        start_date = df.index.min() if days == 'max' else end_date - timedelta(days=days)
        buttons.append(dict(method='relayout', label=label,
                            args=[{'xaxis.range': [start_date, end_date], 'yaxis.autorange': True}]))
    fig.update_layout(title_x=0, yaxis_title="Pontos Percentuais (%)", xaxis_title="Data", showlegend=False,
                      updatemenus=[dict(type="buttons", direction="right", showactive=True, x=1, xanchor="right", y=1.05,
                                        yanchor="bottom", buttons=buttons)])
    start_date_1y = end_date - timedelta(days=365)
    filtered_series = df.loc start_date_1y:end_date, ticker].dropna()
    fig.update_xaxes(range=[start_date_1y, end_date])
    if not filtered_series.empty:
        min_y, max_y = filtered_series.min(), filtered_series.max()
        padding = (max_y - min_y) * 0.10 if (max_y - min_y) > 0 else 0.5
        fig.update_yaxes(range=[min_y - padding, max_y + padding])
    return fig

def gerar_grafico_spread_br_eua(df_br, df_usa):
    df_br.name = 'BR10Y'
    df_usa = df_usa['DGS10']
    df_merged = pd.merge(df_br, df_usa, left_index=True, right_index=True, how='inner')
    df_merged['Spread'] = df_merged['BR10Y'] - df_merged['DGS10']
    fig = px.line(df_merged, y='Spread', title='Spread de Juros 10 Anos: NTN-B (Brasil) vs. Treasury (EUA)',
                    template='plotly_dark')
    end_date = df_merged.index.max()
    buttons = []
    periods = {'1A': 365, '2A': 730, '5A': 1825, 'Máx': 'max'}
    for label, days in periods.items():
        start_date = df_merged.index.min() if days == 'max' else end_date - timedelta(days=days)
        buttons.append(dict(method='relayout', label=label,
                            args=[{'xaxis.range': [start_date, end_date], 'yaxis.autorange': True}]))
    fig.update_layout(
        title_x=0, yaxis_title="Diferença (Pontos Percentuais)", xaxis_title="Data",
        updatemenus=[dict(type="buttons", direction="right", showactive=True, x=1, xanchor="right", y=1.05,
                           yanchor="bottom", buttons=buttons)]
    )
    start_date_1y = end_date - timedelta(days=365)
    filtered_series = df_merged.loc start_date_1y:end_date, 'Spread'].dropna()
    fig.update_xaxes(range=[start_date_1y, end_date])
    if not filtered_series.empty:
        min_y, max_y = filtered_series.min(), filtered_series.max()
        padding = (max_y - min_y) * 0.10 if (max_y - min_y) > 0 else 0.5
        fig.update_yaxes(range=[min_y - padding, max_y + padding])
    return fig

# --- BLOCO 5: LÓGICA DA PÁGINA DE AÇÕES BR ---
@st.cache_data(ttl=3600*24)
def executar_analise_insiders():
    """Função principal que orquestra o download e processamento dos dados de insiders."""
    ANO_ATUAL = datetime.now().year
    URL_MOVIMENTACOES = f"https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/VLMO/DADOS/vlmo_cia_aberta_{ANO_ATUAL}.zip"
    URL_CADASTRO = f"https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FCA/DADOS/fca_cia_aberta_{ANO_ATUAL}.zip"
    ZIP_MOVIMENTACOES, CSV_MOVIMENTACOES = "movimentacoes.zip", f
