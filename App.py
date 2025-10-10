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
import io # Adicionado para a nova funcionalidade
from streamlit_option_menu import option_menu
import pandas_ta as ta
from scipy import stats

# --- CONFIGURAÇÃO GERAL DA PÁGINA ---
st.set_page_config(layout="wide", page_title="MOBBT")

# --- BLOCO 1: LÓGICA DO DASHBOARD DO TESOURO DIRETO ---
@st.cache_data(ttl=3600*4)
def obter_dados_tesouro():
    # ... (código existente inalterado)
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
    # ... (código existente inalterado)
    df_ntnb = df_tesouro[df_tesouro['Tipo Titulo'] == 'Tesouro IPCA+ com Juros Semestrais'].copy()
    if df_ntnb.empty: return pd.Series(dtype=float)
    resultados = {}
    for data_base in df_ntnb['Data Base'].unique():
        df_dia = df_ntnb[df_ntnb['Data Base'] == data_base]
        vencimentos_do_dia = df_dia['Data Vencimento'].unique()
        if len(vencimentos_do_dia) > 0:
            target_10y = pd.to_datetime(data_base) + pd.DateOffset(years=10)
            venc_10y = min(vencimentos_do_dia, key=lambda d: abs(d - target_10y))
            taxa = df_dia[df_dia['Data Vencimento'] == venc_10y]['Taxa Compra Manha'].iloc[0]
            resultados[data_base] = taxa
    return pd.Series(resultados).sort_index()

def gerar_grafico_historico_tesouro(df, tipo, vencimento, metrica='Taxa Compra Manha'):
    # ... (código existente inalterado)
    df_filtrado = df[(df['Tipo Titulo'] == tipo) & (df['Data Vencimento'] == vencimento)].sort_values('Data Base')
    titulo = f'Histórico da Taxa de Compra: {tipo} (Venc. {vencimento.strftime("%d/%m/%Y")})' if metrica == 'Taxa Compra Manha' else f'Histórico do Preço Unitário (PU): {tipo} (Venc. {vencimento.strftime("%d/%m/%Y")})'
    eixo_y = "Taxa de Compra (% a.a.)" if metrica == 'Taxa Compra Manha' else "Preço Unitário (R$)"
    fig = px.line(df_filtrado, x='Data Base', y=metrica, title=titulo, template='plotly_dark')
    fig.update_layout(title_x=0, yaxis_title=eixo_y, xaxis_title="Data")
    return fig
# Adicione esta função nova ao seu código, de preferência no Bloco 1

def gerar_grafico_ntnb_multiplos_vencimentos(df_ntnb_all, vencimentos, metrica):
    """
    Gera um gráfico comparativo para múltiplos vencimentos de NTN-Bs,
    com filtro de tempo e zoom padrão de 5 anos.
    """
    fig = go.Figure()

    if not vencimentos:
        return fig.update_layout(
            title_text="Selecione um ou mais vencimentos para visualizar",
            template="plotly_dark",
            title_x=0.5
        )

    for venc in vencimentos:
        df_venc = df_ntnb_all[df_ntnb_all['Data Vencimento'] == venc].sort_values('Data Base')
        if not df_venc.empty:
            # Extrai o nome do título do primeiro registro (IPCA+ ou IPCA+ com Juros)
            nome_base = df_venc['Tipo Titulo'].iloc[0].replace("Tesouro ", "")
            fig.add_trace(go.Scatter(
                x=df_venc['Data Base'],
                y=df_venc[metrica],
                mode='lines',
                name=f'{nome_base} {venc.year}'
            ))

    titulo = f'Histórico da Taxa de Compra' if metrica == 'Taxa Compra Manha' else f'Histórico do Preço Unitário (PU)'
    eixo_y = "Taxa de Compra (% a.a.)" if metrica == 'Taxa Compra Manha' else "Preço Unitário (R$)"
    
    fig.update_layout(
        title_text=titulo, title_x=0,
        yaxis_title=eixo_y,
        xaxis_title="Data",
        template='plotly_dark',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Adiciona o seletor de range e define o zoom padrão para 5 anos
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1a", step="year", stepmode="backward"),
                dict(count=3, label="3a", step="year", stepmode="backward"),
                dict(count=5, label="5a", step="year", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(step="all", label="Tudo")
            ]),
            bgcolor="#333952",
            font=dict(color="white")
        )
    )
    
    # Define a visualização inicial padrão para os últimos 5 anos
    if not df_ntnb_all.empty:
        end_date = df_ntnb_all['Data Base'].max()
        start_date = end_date - pd.DateOffset(years=5)
        fig.update_xaxes(range=[start_date, end_date])

    return fig
@st.cache_data
def calcular_inflacao_implicita(df):
    # ... (código existente inalterado)
    df_recente = df[df['Data Base'] == df['Data Base'].max()].copy()
    tipos_ipca = ['Tesouro IPCA+ com Juros Semestrais', 'Tesouro IPCA+']
    df_ipca_raw = df_recente[df_recente['Tipo Titulo'].isin(tipos_ipca)]
    df_prefixados = df_recente[df_recente['Tipo Titulo'] == 'Tesouro Prefixado'].set_index('Data Vencimento')
    df_ipca = df_ipca_raw.sort_values('Tipo Titulo', ascending=False).drop_duplicates('Data Vencimento').set_index('Data Vencimento')
    if df_prefixados.empty or df_ipca.empty: return pd.DataFrame()
    inflacao_implicita = []
    for venc_prefixado, row_prefixado in df_prefixados.iterrows():
        venc_ipca_proximo = min(df_ipca.index, key=lambda d: abs(d - venc_prefixado))
        if abs((venc_ipca_proximo - venc_prefixado).days) < 550:
            taxa_prefixada, taxa_ipca = row_prefixado['Taxa Compra Manha'], df_ipca.loc[venc_ipca_proximo]['Taxa Compra Manha']
            breakeven = (((1 + taxa_prefixada / 100) / (1 + taxa_ipca / 100)) - 1) * 100
            inflacao_implicita.append({'Vencimento do Prefixo': venc_prefixado, 'Inflação Implícita (% a.a.)': breakeven})
    if not inflacao_implicita: return pd.DataFrame()
    return pd.DataFrame(inflacao_implicita).sort_values('Vencimento do Prefixo').set_index('Vencimento do Prefixo')

@st.cache_data
def gerar_grafico_spread_juros(df):
    """
    Calcula o spread de juros 10y vs 2y com vencimentos rolantes (dinâmicos).
    Para cada dia no histórico, busca os títulos com vencimentos mais próximos de 2 e 10 anos.
    """
    df_ntnf = df[df['Tipo Titulo'] == 'Tesouro Prefixado com Juros Semestrais'].copy()
    if df_ntnf.empty:
        return go.Figure().update_layout(title_text="Não há dados de Tesouro Prefixado com Juros Semestrais.")

    # Agrupa por data para processamento eficiente
    df_ntnf_grouped = df_ntnf.groupby('Data Base')
    spread_results = []

    for data_base, df_dia in df_ntnf_grouped:
        vencimentos_do_dia = df_dia['Data Vencimento'].unique()

        if len(vencimentos_do_dia) < 2:
            continue

        target_2y = pd.to_datetime(data_base) + pd.DateOffset(years=2)
        target_10y = pd.to_datetime(data_base) + pd.DateOffset(years=10)

        # Encontra os vencimentos mais próximos para a data atual do loop
        venc_curto = min(vencimentos_do_dia, key=lambda d: abs(d - target_2y))
        venc_longo = min(vencimentos_do_dia, key=lambda d: abs(d - target_10y))

        if venc_curto == venc_longo:
            continue

        taxa_curta = df_dia[df_dia['Data Vencimento'] == venc_curto]['Taxa Compra Manha'].iloc[0]
        taxa_longa = df_dia[df_dia['Data Vencimento'] == venc_longo]['Taxa Compra Manha'].iloc[0]
        
        spread = (taxa_longa - taxa_curta) * 100 # Em basis points
        spread_results.append({'Data': data_base, 'Spread': spread})

    if not spread_results:
        return go.Figure().update_layout(title_text="Não foi possível calcular o spread dinâmico.")

    df_spread_final = pd.DataFrame(spread_results).set_index('Data').sort_index()

    # --- Plotagem e Filtros ---
    fig = px.area(df_spread_final, y='Spread', title='Spread de Juros (10 Anos vs. 2 Anos) - Vencimentos Rolantes', template='plotly_dark')
    
    end_date = df_spread_final.index.max()
    start_date_real = df_spread_final.index.min()
    buttons = []
    periods = {'1A': 365, '2A': 730, '5A': 1825, 'Máx': 'max'}
    
    for label, days in periods.items():
        if days == 'max':
            start_date_button = start_date_real
        else:
            start_date_calculada = end_date - timedelta(days=days)
            start_date_button = max(start_date_calculada, start_date_real)
            
        buttons.append(dict(method='relayout', label=label, args=[{'xaxis.range': [start_date_button, end_date], 'yaxis.autorange': True}]))
    
    fig.update_layout(
        title_x=0, 
        yaxis_title="Diferença (Basis Points)", 
        xaxis_title="Data", 
        showlegend=False,
        updatemenus=[dict(
            type="buttons",
            direction="right",
            showactive=True,
            x=1,
            xanchor="right",
            y=1.05,
            yanchor="bottom",
            buttons=buttons
        )]
    )
    
    # --- INÍCIO DA CORREÇÃO DE ESCALA ---
    # Define a visualização inicial padrão para os últimos 5 anos
    start_date_5y_calculada = end_date - pd.DateOffset(years=5)
    start_date_default = max(start_date_5y_calculada, start_date_real)
    fig.update_xaxes(range=[start_date_default, end_date])

    # Filtra o dataframe para o período de visualização padrão
    df_visible = df_spread_final.loc[start_date_default:end_date]

    if not df_visible.empty:
        # Calcula o mínimo e máximo do período visível
        min_y = df_visible['Spread'].min()
        max_y = df_visible['Spread'].max()
        
        # Adiciona uma margem (padding) para não ficar colado nas bordas
        padding = (max_y - min_y) * 0.10
        if padding < 10: # Garante uma margem mínima
            padding = 10
            
        # Define o range do eixo Y para a visualização inicial
        fig.update_yaxes(range=[min_y - padding, max_y + padding])
    # --- FIM DA CORREÇÃO DE ESCALA ---

    return fig


def gerar_grafico_ettj_curto_prazo(df):
    # ... (código existente inalterado)
    df_prefixado = df[df['Tipo Titulo'] == 'Tesouro Prefixado'].copy()
    if df_prefixado.empty: return go.Figure().update_layout(title_text="Não há dados para 'Tesouro Prefixado'.")
    datas_disponiveis = sorted(df_prefixado['Data Base'].unique())
    data_recente = datas_disponiveis[-1]
    targets = {f'Hoje ({data_recente.strftime("%d/%m/%Y")})': data_recente, '1 dia Atrás': data_recente - pd.DateOffset(days=1),'2 dias Atrás': data_recente - pd.DateOffset(days=2),'3 dias Atrás': data_recente - pd.DateOffset(days=3),'4 dias Atrás': data_recente - pd.DateOffset(days=4),'5 dias Atrás': data_recente - pd.DateOffset(days=5)}
    datas_para_plotar = {}
    for legenda_base, data_alvo in targets.items():
        datas_validas = [d for d in datas_disponiveis if d <= data_alvo]
        if datas_validas:
            data_real = max(datas_validas)
            if data_real not in datas_para_plotar.values():
                legenda_final = f'{" ".join(legenda_base.split(" ")[:2])} ({data_real.strftime("%d/%m/%Y")})' if 'Atrás' in legenda_base else legenda_base
                datas_para_plotar[legenda_final] = data_real
    fig = go.Figure()
    for legenda, data_base in datas_para_plotar.items():
        df_data = df_prefixado[df_prefixado['Data Base'] == data_base].sort_values('Data Vencimento')
        df_data['Dias Uteis'] = np.busday_count(df_data['Data Base'].values.astype('M8[D]'), df_data['Data Vencimento'].values.astype('M8[D]'))
        line_style = dict(dash='dash') if not legenda.startswith('Hoje') else {}
        fig.add_trace(go.Scatter(x=df_data['Dias Uteis'], y=df_data['Taxa Compra Manha'], mode='lines+markers', name=legenda, line=line_style))
    fig.update_layout(title_text='Curva de Juros (ETTJ) - Curto Prazo (últimos 5 dias)', title_x=0, xaxis_title='Dias Úteis até o Vencimento', yaxis_title='Taxa (% a.a.)', template='plotly_dark', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def gerar_grafico_ettj_longo_prazo(df):
    # ... (código existente inalterado)
    df_prefixado = df[df['Tipo Titulo'] == 'Tesouro Prefixado'].copy()
    if df_prefixado.empty: return go.Figure().update_layout(title_text="Não há dados para 'Tesouro Prefixado'.")
    datas_disponiveis = sorted(df_prefixado['Data Base'].unique())
    data_recente = datas_disponiveis[-1]
    targets = {f'Hoje ({data_recente.strftime("%d/%m/%Y")})': data_recente, '1 Semana Atrás': data_recente - pd.DateOffset(weeks=1), '1 Mês Atrás': data_recente - pd.DateOffset(months=1), '3 Meses Atrás': data_recente - pd.DateOffset(months=3), '6 Meses Atrás': data_recente - pd.DateOffset(months=6), '1 Ano Atrás': data_recente - pd.DateOffset(years=1)}
    datas_para_plotar = {}
    for legenda_base, data_alvo in targets.items():
        datas_validas = [d for d in datas_disponiveis if d <= data_alvo]
        if datas_validas:
            data_real = max(datas_validas)
            if data_real not in datas_para_plotar.values():
                legenda_final = f'{" ".join(legenda_base.split(" ")[:2])} ({data_real.strftime("%d/%m/%Y")})' if not legenda_base.startswith('Hoje') else legenda_base
                datas_para_plotar[legenda_final] = data_real
    fig = go.Figure()
    for legenda, data_base in datas_para_plotar.items():
        df_data = df_prefixado[df_prefixado['Data Base'] == data_base].sort_values('Data Vencimento')
        df_data['Dias Uteis'] = np.busday_count(df_data['Data Base'].values.astype('M8[D]'), df_data['Data Vencimento'].values.astype('M8[D]'))
        line_style = dict(dash='dash') if not legenda.startswith('Hoje') else {}
        fig.add_trace(go.Scatter(x=df_data['Dias Uteis'], y=df_data['Taxa Compra Manha'], mode='lines+markers', name=legenda, line=line_style))
    fig.update_layout(title_text='Curva de Juros (ETTJ) - Longo Prazo (Comparativo Histórico)', title_x=0, xaxis_title='Dias Úteis até o Vencimento', yaxis_title='Taxa (% a.a.)', template='plotly_dark', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

# --- BLOCO 2: LÓGICA DO DASHBOARD DE INDICADORES ECONÔMICOS ---
@st.cache_data(ttl=3600*4)
def carregar_dados_bcb():
    # ... (código existente inalterado)
    SERIES_CONFIG = {'Spread Bancário': {'id': 20783}, 'Inadimplência': {'id': 21082}, 'Crédito/PIB': {'id': 20622}, 'Juros Médio': {'id': 20714}, 'Confiança Consumidor': {'id': 4393}, 'IPCA': {'id': 16122}, 'Atraso 15-90d Total': {'id': 21006}, 'Atraso 15-90d Agro': {'id': 21069}, 'Inadimplência Crédito Rural': {'id': 21146}}
    lista_dfs_sucesso, config_sucesso = [], {}
    for name, config in SERIES_CONFIG.items():
        try:
            df_temp = sgs.get({name: config['id']}, start='2010-01-01'); lista_dfs_sucesso.append(df_temp); config_sucesso[name] = config
        except Exception as e: st.warning(f"Não foi possível carregar o indicador '{name}': {e}")
    if not lista_dfs_sucesso: return pd.DataFrame(), {}
    df_full = pd.concat(lista_dfs_sucesso, axis=1); df_full.ffill(inplace=True); df_full.dropna(inplace=True)
    return df_full, config_sucesso

# --- BLOCO 3: LÓGICA DO DASHBOARD DE COMMODITIES ---
@st.cache_data(ttl=3600*4)
def carregar_dados_commodities():
    # ... (código existente inalterado)
    commodities_map = {'Petróleo Brent': 'BZ=F', 'Cacau': 'CC=F', 'Petróleo WTI': 'CL=F', 'Algodão': 'CT=F', 'Ouro': 'GC=F', 'Cobre': 'HG=F', 'Óleo de Aquecimento': 'HO=F', 'Café': 'KC=F', 'Trigo (KC HRW)': 'KE=F', 'Madeira': 'LBS=F', 'Gado Bovino': 'LE=F', 'Gás Natural': 'NG=F', 'Suco de Laranja': 'OJ=F', 'Paládio': 'PA=F', 'Platina': 'PL=F', 'Gasolina RBOB': 'RB=F', 'Açúcar': 'SB=F', 'Prata': 'SI=F', 'Milho': 'ZC=F', 'Óleo de Soja': 'ZL=F', 'Aveia': 'ZO=F', 'Arroz': 'ZR=F', 'Soja': 'ZS=F'}
    dados_commodities_raw = {}
    with st.spinner("Baixando dados históricos de commodities... (cache de 4h)"):
        for nome, ticker in commodities_map.items():
            try:
                dado = yf.download(ticker, period='max', auto_adjust=True, progress=False)
                if not dado.empty: dados_commodities_raw[nome] = dado['Close']
            except Exception: pass
    categorized_commodities = {'Energia': ['Petróleo Brent', 'Petróleo WTI', 'Óleo de Aquecimento', 'Gás Natural', 'Gasolina RBOB'], 'Metais Preciosos': ['Ouro', 'Paládio', 'Platina', 'Prata'], 'Metais Industriais': ['Cobre'], 'Agricultura': ['Cacau', 'Algodão', 'Café', 'Trigo (KC HRW)', 'Madeira', 'Gado Bovino', 'Suco de Laranja', 'Açúcar', 'Milho', 'Óleo de Soja', 'Aveia', 'Arroz', 'Soja']}
    dados_por_categoria = {}
    for categoria, nomes in categorized_commodities.items():
        series_da_categoria = {nome: dados_commodities_raw[nome] for nome in nomes if nome in dados_commodities_raw}
        if series_da_categoria:
            df_cat = pd.concat(series_da_categoria, axis=1); df_cat.columns = series_da_categoria.keys()
            dados_por_categoria[categoria] = df_cat
    return dados_por_categoria

def calcular_variacao_commodities(dados_por_categoria):
    # ... (código existente inalterado)
    all_series = [s for df in dados_por_categoria.values() for s in [df[col].dropna() for col in df.columns]]
    if not all_series: return pd.DataFrame()
    df_full = pd.concat(all_series, axis=1); df_full.sort_index(inplace=True)
    if df_full.empty: return pd.DataFrame()
    latest_date = df_full.index.max()
    latest_prices = df_full.loc[latest_date]
    periods = {'1 Dia': 1, '1 Semana': 7, '1 Mês': 30, '3 Meses': 91, '6 Meses': 182, '1 Ano': 365}
    results = []
    for name in df_full.columns:
        res = {'Commodity': name, 'Preço Atual': latest_prices[name]}; series = df_full[name].dropna()
        for label, days in periods.items():
            past_date = latest_date - timedelta(days=days); past_price = series.asof(past_date)
            res[f'Variação {label}'] = ((latest_prices[name] - past_price) / past_price) if pd.notna(past_price) and past_price > 0 else np.nan
        results.append(res)
    return pd.DataFrame(results).set_index('Commodity')

def colorir_negativo_positivo(val):
    # ... (código existente inalterado)
    if pd.isna(val) or val == 0: return ''
    return f"color: {'#4CAF50' if val > 0 else '#F44336'}"

def gerar_dashboard_commodities(dados_preco_por_categoria):
    # ... (código existente inalterado)
    all_commodity_names = [name for df in dados_preco_por_categoria.values() for name in df.columns]
    total_subplots = len(all_commodity_names)
    if total_subplots == 0: return go.Figure().update_layout(title_text="Nenhum dado de commodity disponível.")
    num_cols, num_rows = 4, int(np.ceil(total_subplots / 4))
    fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=all_commodity_names)
    idx = 0
    for df_cat in dados_preco_por_categoria.values():
        for commodity_name in df_cat.columns:
            row, col = (idx // num_cols) + 1, (idx % num_cols) + 1
            fig.add_trace(go.Scatter(x=df_cat.index, y=df_cat[commodity_name], mode='lines', name=commodity_name), row=row, col=col)
            idx += 1
    end_date = datetime.now(); buttons = []; 
    periods = {'1M': 30, '3M': 91, '6M': 182, 'YTD': 'ytd', '1A': 365, '5A': 365*5, '10A': 3650, 'Máx': 'max'}
    for label, days in periods.items():
        if days == 'ytd': start_date = datetime(end_date.year, 1, 1)
        elif days == 'max': start_date = min([df.index.min() for df in dados_preco_por_categoria.values() if not df.empty])
        else: start_date = end_date - timedelta(days=days)
        update_args = {}
        for i in range(1, total_subplots + 1):
            update_args[f'xaxis{i if i > 1 else ""}.range'], update_args[f'yaxis{i if i > 1 else ""}.autorange'] = [start_date, end_date], True
        buttons.append(dict(method='relayout', label=label, args=[update_args]))
    active_button_index = list(periods.keys()).index('1A') if '1A' in list(periods.keys()) else 4
    fig.update_layout(title_text="Dashboard de Preços Históricos de Commodities", title_x=0, template="plotly_dark", height=250 * num_rows, showlegend=False,
                        updatemenus=[dict(type="buttons", direction="right", showactive=True, x=1, xanchor="right", y=1.05, yanchor="bottom", buttons=buttons, active=active_button_index)])
    start_date_1y = end_date - timedelta(days=365); idx = 0
    for df_cat in dados_preco_por_categoria.values():
        for i, commodity_name in enumerate(df_cat.columns, start=idx):
            fig.layout[f'xaxis{i+1 if i+1 > 1 else ""}.range'] = [start_date_1y, end_date]
            series = df_cat[commodity_name]; filtered_series = series[(series.index >= start_date_1y) & (series.index <= end_date)].dropna()
            if not filtered_series.empty:
                min_y, max_y = filtered_series.min(), filtered_series.max(); padding = (max_y - min_y) * 0.05
                fig.layout[f'yaxis{i+1 if i+1 > 1 else ""}.range'] = [min_y - padding, max_y + padding]
            else: fig.layout[f'yaxis{i+1 if i+1 > 1 else ""}.autorange'] = True
        idx += len(df_cat.columns)
    return fig

# --- BLOCO 4: LÓGICA DO DASHBOARD DE INDICADORES INTERNACIONAIS ---
@st.cache_data(ttl=3600*4)
def carregar_dados_fred(api_key, tickers_dict):
    # ... (código existente inalterado)
    fred = Fred(api_key=api_key)
    lista_series = []
    st.info("Carregando dados do FRED... (Cache de 4h)")
    for ticker in tickers_dict.keys():
        try:
            serie = fred.get_series(ticker); serie.name = ticker; lista_series.append(serie)
        except Exception as e: st.warning(f"Não foi possível carregar o ticker '{ticker}' do FRED: {e}")
    if not lista_series: return pd.DataFrame()
    return pd.concat(lista_series, axis=1).ffill()

def gerar_grafico_fred(df, ticker, titulo):
    # ... (código existente inalterado)
    if ticker not in df.columns or df[ticker].isnull().all():
        return go.Figure().update_layout(title_text=f"Dados para {ticker} não encontrados.")
    fig = px.line(df, y=ticker, title=titulo, template='plotly_dark')
    if ticker == 'T10Y2Y':
        fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Inversão", annotation_position="bottom right")
    end_date = df.index.max()
    buttons = []
    periods = {'6M': 182, '1A': 365, '2A': 730, '5A': 1825, '10A': 3650, 'Máx': 'max'}
    for label, days in periods.items():
        start_date = df.index.min() if days == 'max' else end_date - timedelta(days=days)
        buttons.append(dict(method='relayout', label=label, args=[{'xaxis.range': [start_date, end_date], 'yaxis.autorange': True}]))
    fig.update_layout(title_x=0, yaxis_title="Pontos Percentuais (%)", xaxis_title="Data", showlegend=False,
                        updatemenus=[dict(type="buttons", direction="right", showactive=True, x=1, xanchor="right", y=1.05, yanchor="bottom", buttons=buttons)])
    start_date_1y = end_date - timedelta(days=365)
    filtered_series = df.loc[start_date_1y:end_date, ticker].dropna()
    fig.update_xaxes(range=[start_date_1y, end_date])
    if not filtered_series.empty:
        min_y, max_y = filtered_series.min(), filtered_series.max()
        padding = (max_y - min_y) * 0.10 if (max_y - min_y) > 0 else 0.5
        fig.update_yaxes(range=[min_y - padding, max_y + padding])
    return fig

def gerar_grafico_spread_br_eua(df_br, df_usa):
    # ... (código existente inalterado)
    df_br.name = 'BR10Y'
    df_usa = df_usa['DGS10']
    df_merged = pd.merge(df_br, df_usa, left_index=True, right_index=True, how='inner')
    df_merged['Spread'] = df_merged['BR10Y'] - df_merged['DGS10']
    fig = px.line(df_merged, y='Spread', title='Spread de Juros 10 Anos: NTN-B (Brasil) vs. Treasury (EUA)', template='plotly_dark')
    end_date = df_merged.index.max()
    buttons = []
    periods = {'1A': 365, '2A': 730, '5A': 1825, 'Máx': 'max'}
    for label, days in periods.items():
        start_date = df_merged.index.min() if days == 'max' else end_date - timedelta(days=days)
        buttons.append(dict(method='relayout', label=label, args=[{'xaxis.range': [start_date, end_date], 'yaxis.autorange': True}]))
    fig.update_layout(
        title_x=0, yaxis_title="Diferença (Pontos Percentuais)", xaxis_title="Data",
        updatemenus=[dict(type="buttons", direction="right", showactive=True, x=1, xanchor="right", y=1.05, yanchor="bottom", buttons=buttons)]
    )
    start_date_1y = end_date - timedelta(days=365)
    filtered_series = df_merged.loc[start_date_1y:end_date, 'Spread'].dropna()
    fig.update_xaxes(range=[start_date_1y, end_date])
    if not filtered_series.empty:
        min_y, max_y = filtered_series.min(), filtered_series.max()
        padding = (max_y - min_y) * 0.10 if (max_y - min_y) > 0 else 0.5
        fig.update_yaxes(range=[min_y - padding, max_y + padding])
    return fig

# --- BLOCO 5: LÓGICA DA PÁGINA DE AÇÕES BR ---
@st.cache_data(ttl=3600*24)
def executar_analise_insiders():
    # ... (código existente inalterado)
    """Função principal que orquestra o download e processamento dos dados de insiders."""
    ANO_ATUAL = datetime.now().year
    URL_MOVIMENTACOES = f"https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/VLMO/DADOS/vlmo_cia_aberta_{ANO_ATUAL}.zip"
    URL_CADASTRO = f"https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FCA/DADOS/fca_cia_aberta_{ANO_ATUAL}.zip"
    ZIP_MOVIMENTACOES, CSV_MOVIMENTACOES = "movimentacoes.zip", f"vlmo_cia_aberta_con_{ANO_ATUAL}.csv"
    ZIP_CADASTRO, CSV_CADASTRO = "cadastro.zip", f"fca_cia_aberta_valor_mobiliario_{ANO_ATUAL}.csv"
    
    def _cvm_baixar_zip(url, nome_zip, nome_csv):
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            with open(nome_zip, 'wb') as f: f.write(response.content)
            with zipfile.ZipFile(nome_zip, 'r') as z: z.extract(nome_csv)
            os.remove(nome_zip)
            return nome_csv
        except Exception as e:
            st.error(f"Erro no download de {url}: {e}")
            if os.path.exists(nome_zip): os.remove(nome_zip)
            return None

    def _obter_market_cap_individual(ticker):
        if pd.isna(ticker) or not isinstance(ticker, str): return ticker, np.nan
        try:
            stock = yf.Ticker(f"{ticker.strip()}.SA")
            return ticker, stock.info.get('marketCap', np.nan)
        except Exception:
            return ticker, np.nan

    caminho_csv_mov = _cvm_baixar_zip(URL_MOVIMENTACOES, ZIP_MOVIMENTACOES, CSV_MOVIMENTACOES)
    caminho_csv_cad = _cvm_baixar_zip(URL_CADASTRO, ZIP_CADASTRO, CSV_CADASTRO)

    if not caminho_csv_mov or not caminho_csv_cad: return None, None, None

    df_mov = pd.read_csv(caminho_csv_mov, sep=';', encoding='ISO-8859-1', on_bad_lines='skip')
    df_cad = pd.read_csv(caminho_csv_cad, sep=';', encoding='ISO-8859-1', on_bad_lines='skip', usecols=['CNPJ_Companhia', 'Codigo_Negociacao'])
    os.remove(caminho_csv_mov); os.remove(caminho_csv_cad)

    df_mov['Data_Movimentacao'] = pd.to_datetime(df_mov['Data_Movimentacao'], errors='coerce')
    df_mov.dropna(subset=['Data_Movimentacao'], inplace=True)
    df_mov = df_mov[df_mov['Tipo_Movimentacao'].isin(['Compra à vista', 'Venda à vista'])]
    ultimo_mes = df_mov['Data_Movimentacao'].max().to_period('M')
    df_mes = df_mov[df_mov['Data_Movimentacao'].dt.to_period('M') == ultimo_mes].copy()
    df_mes['Volume_Net'] = np.where(df_mes['Tipo_Movimentacao'] == 'Compra à vista', df_mes['Volume'], -df_mes['Volume'])

    df_controladores = df_mes[df_mes['Tipo_Cargo'] == 'Controlador ou Vinculado'].copy()
    df_outros = df_mes[df_mes['Tipo_Cargo'] != 'Controlador ou Vinculado'].copy()
    
    df_net_controladores = df_controladores.groupby(['CNPJ_Companhia', 'Nome_Companhia'])['Volume_Net'].sum().reset_index()
    df_net_outros = df_outros.groupby(['CNPJ_Companhia', 'Nome_Companhia'])['Volume_Net'].sum().reset_index()

    cnpjs_unicos = pd.concat([df_net_controladores[['CNPJ_Companhia']], df_net_outros[['CNPJ_Companhia']]]).drop_duplicates()
    df_tickers = df_cad.dropna().drop_duplicates(subset=['CNPJ_Companhia'])
    df_lookup = pd.merge(cnpjs_unicos, df_tickers, on='CNPJ_Companhia', how='left')
    
    market_caps = {}
    tickers_para_buscar = df_lookup['Codigo_Negociacao'].dropna().unique().tolist()
    progress_bar = st.progress(0, text="Buscando valores de mercado...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {executor.submit(_obter_market_cap_individual, ticker): ticker for ticker in tickers_para_buscar}
        for i, future in enumerate(as_completed(future_to_ticker)):
            ticker, market_cap = future.result()
            market_caps[ticker] = market_cap
            progress_bar.progress((i + 1) / len(tickers_para_buscar), text=f"Buscando valores de mercado... ({i+1}/{len(tickers_para_buscar)})")
    progress_bar.empty()
    df_market_caps = pd.DataFrame(list(market_caps.items()), columns=['Codigo_Negociacao', 'MarketCap'])
    df_market_cap_lookup = pd.merge(df_lookup, df_market_caps, on="Codigo_Negociacao", how="left")
    
    df_final_controladores = pd.merge(df_net_controladores, df_market_cap_lookup, on='CNPJ_Companhia', how='left')
    df_final_controladores['Volume_vs_MarketCap_Pct'] = (df_final_controladores['Volume_Net'] / df_final_controladores['MarketCap']) * 100
    df_final_controladores.fillna({'Volume_vs_MarketCap_Pct': 0}, inplace=True)

    df_final_outros = pd.merge(df_net_outros, df_market_cap_lookup, on='CNPJ_Companhia', how='left')
    df_final_outros['Volume_vs_MarketCap_Pct'] = (df_final_outros['Volume_Net'] / df_final_outros['MarketCap']) * 100
    df_final_outros.fillna({'Volume_vs_MarketCap_Pct': 0}, inplace=True)
    
    return df_final_controladores, df_final_outros, ultimo_mes

def gerar_graficos_insiders_plotly(df_dados, top_n=10):
    # ... (código existente inalterado)
    if df_dados.empty: return None, None

    # Gráfico 1: Volume
    df_plot_volume = df_dados.sort_values(by='Volume_Net', ascending=True).tail(top_n)
    fig_volume = px.bar(
        df_plot_volume,
        y='Nome_Companhia',
        x='Volume_Net',
        orientation='h',
        title=f'Top {top_n} por Volume Líquido',
        template='plotly_dark',
        text='Volume_Net'
    )
    fig_volume.update_traces(texttemplate='R$ %{text:,.2s}', textposition='outside')
    fig_volume.update_layout(title_x=0, xaxis_title="Volume Líquido (R$)", yaxis_title="")

    # Gráfico 2: Relevância
    df_plot_relevancia = df_dados.sort_values(by='Volume_vs_MarketCap_Pct', ascending=True).tail(top_n)
    fig_relevancia = px.bar(
        df_plot_relevancia,
        y='Nome_Companhia',
        x='Volume_vs_MarketCap_Pct',
        orientation='h',
        title=f'Top {top_n} por Relevância (Volume / Valor de Mercado)',
        template='plotly_dark',
        text='Volume_vs_MarketCap_Pct'
    )
    fig_relevancia.update_traces(texttemplate='%{text:.3f}%', textposition='outside')
    fig_relevancia.update_layout(title_x=0, xaxis_title="Volume como % do Valor de Mercado", yaxis_title="")
    
    return fig_volume, fig_relevancia

@st.cache_data
def carregar_dados_acoes(tickers, period="max"):
    # ... (código existente inalterado)
    try:
        data = yf.download(tickers, period=period, auto_adjust=True)['Close']
        if isinstance(data, pd.Series): 
            data = data.to_frame(tickers[0])
        return data.dropna()
    except Exception:
        return pd.DataFrame()

@st.cache_data
def calcular_metricas_ratio(data, ticker_a, ticker_b, window=252):
    # ... (código existente inalterado)
    ratio = data[ticker_a] / data[ticker_b]
    df_metrics = pd.DataFrame({'Ratio': ratio})
    df_metrics['Rolling_Mean'] = ratio.rolling(window=window).mean()
    rolling_std = ratio.rolling(window=window).std()
    static_median = ratio.median()
    static_std = ratio.std()
    df_metrics['Static_Median'] = static_median
    df_metrics['Upper_Band_2x_Rolling'] = df_metrics['Rolling_Mean'] + (2 * rolling_std)
    df_metrics['Lower_Band_2x_Rolling'] = df_metrics['Rolling_Mean'] - (2 * rolling_std)
    df_metrics['Upper_Band_1x_Static'] = static_median + (1 * static_std)
    df_metrics['Lower_Band_1x_Static'] = static_median - (1 * static_std)
    df_metrics['Upper_Band_2x_Static'] = static_median + (2 * static_std)
    df_metrics['Lower_Band_2x_Static'] = static_median - (2 * static_std)
    return df_metrics

def calcular_kpis_ratio(df_metrics):
    # ... (código existente inalterado)
    if 'Ratio' not in df_metrics or df_metrics['Ratio'].dropna().empty: return None
    ratio_series = df_metrics['Ratio'].dropna()
    kpis = {"atual": ratio_series.iloc[-1], "media": ratio_series.mean(), "minimo": ratio_series.min(), "data_minimo": ratio_series.idxmin(), "maximo": ratio_series.max(), "data_maximo": ratio_series.idxmax()}
    if kpis["atual"] > 0: kpis["variacao_para_media"] = (kpis["media"] / kpis["atual"] - 1) * 100
    else: kpis["variacao_para_media"] = np.inf
    return kpis

def gerar_grafico_ratio(df_metrics, ticker_a, ticker_b, window):
    # ... (código existente inalterado)
    fig = go.Figure()
    static_median_val = df_metrics['Static_Median'].iloc[-1]
    fig.add_hline(y=static_median_val, line_color='red', line_dash='dash', annotation_text=f'Mediana ({static_median_val:.2f})', annotation_position="top left")
    fig.add_hline(y=df_metrics['Upper_Band_1x_Static'].iloc[-1], line_color='#2ca02c', line_dash='dot', annotation_text='+1 DP Estático', annotation_position="top left")
    fig.add_hline(y=df_metrics['Lower_Band_1x_Static'].iloc[-1], line_color='#2ca02c', line_dash='dot', annotation_text='-1 DP Estático', annotation_position="top left")
    fig.add_hline(y=df_metrics['Upper_Band_2x_Static'].iloc[-1], line_color='#d62728', line_dash='dot', annotation_text='+2 DP Estático', annotation_position="top left")
    fig.add_hline(y=df_metrics['Lower_Band_2x_Static'].iloc[-1], line_color='#d62728', line_dash='dot', annotation_text='-2 DP Estático', annotation_position="top left")
    fig.add_trace(go.Scatter(x=df_metrics.index, y=df_metrics['Upper_Band_2x_Rolling'], mode='lines', line_color='gray', line_width=1, name='Bollinger Superior', showlegend=False))
    fig.add_trace(go.Scatter(x=df_metrics.index, y=df_metrics['Lower_Band_2x_Rolling'], mode='lines', line_color='gray', line_width=1, name='Bollinger Inferior', fill='tonexty', fillcolor='rgba(128,128,128,0.1)', showlegend=False))
    fig.add_trace(go.Scatter(x=df_metrics.index, y=df_metrics['Rolling_Mean'], mode='lines', line_color='orange', line_dash='dash', name=f'Média Móvel ({window}d)'))
    fig.add_trace(go.Scatter(x=df_metrics.index, y=df_metrics['Ratio'], mode='lines', line_color='#636EFA', name='Ratio Atual', line_width=2.5))
    fig.update_layout(title_text=f'Análise de Ratio: {ticker_a} / {ticker_b}', template='plotly_dark', title_x=0, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

# --- BLOCO 6: LÓGICA DO INDICADOR IDEX JGP (NOVO) ---
@st.cache_data(ttl=3600*4) # Cache de 4 horas
def carregar_dados_idex():
    """
    Baixa e processa os dados do IDEX JGP para os índices Geral e Low Rated.
    (Versão estável original)
    """
    st.info("Carregando dados do IDEX JGP... (Cache de 4h)")
    url_geral = "https://jgp-credito-public-s3.s3.us-east-1.amazonaws.com/idex/idex_cdi_geral_datafile.xlsx"
    url_low_rated = "https://jgp-credito-public-s3.s3.us-east-1.amazonaws.com/idex/idex_cdi_low_rated_datafile.xlsx"
    emissores_para_remover = ['AMERICANAS SA', 'Light - Servicos de Eletricidade', 'Aeris', 'Viveo']

    def _processar_url(url):
        response = requests.get(url)
        response.raise_for_status()
        df = pd.read_excel(io.BytesIO(response.content), sheet_name='Detalhado')
        df.columns = df.columns.str.strip()
        df_filtrado = df[~df['Emissor'].isin(emissores_para_remover)].copy()
        df_filtrado['Data'] = pd.to_datetime(df_filtrado['Data'])
        df_filtrado['weighted_spread'] = df_filtrado['Peso no índice (%)'] * df_filtrado['Spread de compra (%)']
        
        daily_spread = df_filtrado.groupby('Data').apply(
            lambda x: x['weighted_spread'].sum() / x['Peso no índice (%)'].sum() if x['Peso no índice (%)'].sum() != 0 else 0
        ).reset_index(name='spread')
        
        return daily_spread.set_index('Data')

    try:
        spread_geral = _processar_url(url_geral)
        spread_low_rated = _processar_url(url_low_rated)
        
        df_final = pd.merge(spread_geral, spread_low_rated, on='Data', how='outer', suffixes=('_geral', '_low_rated'))
        df_final.rename(columns={'spread_geral': 'IDEX Geral (Filtrado)', 'spread_low_rated': 'IDEX Low Rated (Filtrado)'}, inplace=True)
        return df_final.sort_index()

    except Exception as e:
        st.error(f"Erro ao carregar dados do IDEX JGP: {e}")
        return pd.DataFrame()
# --- INÍCIO DO NOVO BLOCO DE CÓDIGO (PARA O BLOCO 6) ---

@st.cache_data(ttl=3600*4) # Cache de 4 horas
def carregar_dados_idex_infra():
    """
    Baixa e processa os dados do IDEX INFRA JGP.
    """
    st.info("Carregando dados do IDEX INFRA... (Cache de 4h)")
    url_infra = "https://jgp-credito-public-s3.s3.us-east-1.amazonaws.com/idex/idex_infra_geral_datafile.xlsx"
    
    try:
        response = requests.get(url_infra)
        response.raise_for_status()
        
        # Lê a planilha 'Detalhado'
        df = pd.read_excel(io.BytesIO(response.content), sheet_name='Detalhado')
        df.columns = df.columns.str.strip()
        
        # Converte a data e calcula o spread ponderado
        df['Data'] = pd.to_datetime(df['Data'])
        df['weighted_spread'] = df['Peso no índice (%)'] * df['MID spread (Bps/NTNB)']
        
        # Agrupa por data para calcular o spread médio diário
        daily_spread = df.groupby('Data').apply(
            lambda x: x['weighted_spread'].sum() / x['Peso no índice (%)'].sum() if x['Peso no índice (%)'].sum() != 0 else 0
        ).reset_index(name='spread_bps_ntnb')
        
        return daily_spread.set_index('Data').sort_index()

    except Exception as e:
        st.error(f"Erro ao carregar dados do IDEX INFRA: {e}")
        return pd.DataFrame()

def gerar_grafico_idex_infra(df_idex_infra):
    """
    Gera um gráfico Plotly para o spread do IDEX INFRA.
    """
    if df_idex_infra.empty:
        return go.Figure().update_layout(title_text="Não foi possível gerar o gráfico do IDEX INFRA.")

    fig = px.line(
        df_idex_infra,
        y='spread_bps_ntnb',
        title='Histórico do Spread Médio Ponderado: IDEX INFRA',
        template='plotly_dark'
    )
    
    # Atualiza os eixos e a legenda
    fig.update_layout(
        title_x=0,
        yaxis_title='Spread Médio (Bps sobre NTNB)',
        xaxis_title='Data',
        showlegend=False
    )
    return fig

# --- FIM DO NOVO BLOCO DE CÓDIGO ---


def gerar_grafico_idex(df_idex):
    """
    Gera um gráfico Plotly comparando os spreads do IDEX Geral e Low Rated.
    (Versão estável original)
    """
    if df_idex.empty:
        return go.Figure().update_layout(title_text="Não foi possível gerar o gráfico do IDEX.")

    fig = px.line(
        df_idex,
        y=['IDEX Geral (Filtrado)', 'IDEX Low Rated (Filtrado)'],
        title='Histórico do Spread Médio Ponderado: IDEX JGP',
        template='plotly_dark'
    )

    fig.update_yaxes(tickformat=".2%")
    fig.update_traces(hovertemplate='%{y:.2%}')

    fig.update_layout(
        title_x=0,
        yaxis_title='Spread Médio Ponderado (%)',
        xaxis_title='Data',
        legend_title_text='Índice',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig
# --- FIM DO NOVO BLOCO ---

# --- CONSTRUÇÃO DA INTERFACE (LAYOUT FINAL COM OPTION_MENU) ---

# --- Lógica para carregar os dados principais uma vez ---
df_tesouro = obter_dados_tesouro()

# --- Configuração do Sidebar com o novo menu ---
with st.sidebar:
    st.title("MOBBT")
    st.caption(f"Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    
    pagina_selecionada = option_menu(
        menu_title="Monitoramento",
        options=[
            "NTN-Bs",
            "Curva de Juros",
            "Crédito Privado",
            "Amplitude", # <-- ADICIONADO
            "Econômicos BR",
            "Commodities",
            "Internacional",
            "Ações BR",
        ],
        # Ícones da https://icons.getbootstrap.com/
        icons=[
            "star-fill",
            "graph-up-arrow",
            "wallet2",
            "water", # <-- ADICIONADO
            "bar-chart-line-fill",
            "box-seam",
            "globe-americas",
            "kanban-fill",
        ],
        menu_icon="speedometer2",
        default_index=0,
        styles={
            "container": {"padding": "5px !important", "background-color": "transparent"},
            "icon": {"color": "#636EFA", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#262830"},
            "nav-link-selected": {"background-color": "#333952"}, # Verde para destacar
        }
    )

# --- Roteamento de Páginas (com nomes atualizados) ---

if pagina_selecionada == "NTN-Bs":
    st.header("Dashboard de Análise de NTN-Bs (Tesouro IPCA+)")
    st.markdown("---")

    if not df_tesouro.empty:
        # --- PAINEL DE ANÁLISE HISTÓRICA (ATUALIZADO) ---
        st.subheader("Análise Histórica Comparativa")
        st.info("Selecione um ou mais vencimentos para comparar a variação da taxa ou preço ao longo do tempo.")
        
        # Filtra apenas os títulos NTN-B
        tipos_ntnb = ['Tesouro IPCA+', 'Tesouro IPCA+ com Juros Semestrais']
        df_ntnb_all = df_tesouro[df_tesouro['Tipo Titulo'].isin(tipos_ntnb)]
        
        # Prepara as opções para o multiselect
        vencimentos_disponiveis = sorted(df_ntnb_all['Data Vencimento'].unique())
        
        # Encontra os vencimentos padrão (2030, 2035, etc.) que realmente existem nos dados
        anos_padrao = [2030, 2035, 2040, 2045, 2060]
        vencimentos_padrao = [v for v in vencimentos_disponiveis if pd.to_datetime(v).year in anos_padrao]

        # Cria os widgets de filtro
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            vencimentos_selecionados = st.multiselect(
                "Selecione os Vencimentos",
                options=vencimentos_disponiveis,
                default=vencimentos_padrao,
                format_func=lambda dt: pd.to_datetime(dt).strftime('%d/%m/%Y'),
                key='multi_venc_ntnb'
            )
        with col2:
             metrica_escolhida = st.radio(
                "Analisar por:", ('Taxa', 'PU'), # Nomes mais curtos para caber
                horizontal=True, key='metrica_ntnb',
                help="Analisar por Taxa de Compra ou Preço Unitário (PU)"
            )
        coluna_metrica = 'Taxa Compra Manha' if metrica_escolhida == 'Taxa' else 'PU Compra Manha'

        # Gera e exibe o gráfico
        fig_hist_ntnb = gerar_grafico_ntnb_multiplos_vencimentos(
            df_ntnb_all, vencimentos_selecionados, metrica=coluna_metrica
        )
        st.plotly_chart(fig_hist_ntnb, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- DEMAIS GRÁFICOS DO DASHBOARD (sem alteração) ---
        bottom_left, bottom_right = st.columns((1, 1))
        with bottom_left:
            st.subheader("Inflação Implícita (Breakeven)")
            df_breakeven = calcular_inflacao_implicita(df_tesouro)
            if not df_breakeven.empty:
                fig_breakeven = px.bar(df_breakeven, y='Inflação Implícita (% a.a.)', text_auto='.2f', title='Inflação Implícita por Vencimento').update_traces(textposition='outside')
                fig_breakeven.update_layout(title_x=0, template='plotly_dark')
                st.plotly_chart(fig_breakeven, use_container_width=True)
            else:
                st.warning("Não há pares de títulos para calcular a inflação implícita hoje.")
        with bottom_right:
            st.subheader("Spread de Juros: Brasil vs. EUA")
            st.info("Diferença entre a taxa da NTN-B de ~10 anos e o título americano de 10 anos.")
            FRED_API_KEY = 'd78668ca6fc142a1248f7cb9132916b0'
            df_fred_br_tab = carregar_dados_fred(FRED_API_KEY, {'DGS10': 'Juros 10 Anos EUA'})
            if not df_fred_br_tab.empty:
                df_juro_br = calcular_juro_10a_br(df_tesouro)
                if not df_juro_br.empty:
                    fig_spread_br_eua = gerar_grafico_spread_br_eua(df_juro_br, df_fred_br_tab)
                    st.plotly_chart(fig_spread_br_eua, use_container_width=True, config={'modeBarButtonsToRemove': ['autoscale']})
                else:
                    st.warning("Não foi possível calcular a série de juros de 10 anos para o Brasil.")
            else:
                st.warning("Não foi possível carregar os dados de juros dos EUA.")
    else:
        st.warning("Não foi possível carregar os dados do Tesouro Direto para exibir esta página.")


elif pagina_selecionada == "Curva de Juros":
    st.header("Estrutura a Termo da Taxa de Juros (ETTJ)")
    st.info("Esta página foca na análise dos títulos públicos prefixados (LTNs e NTN-Fs), que formam a curva de juros nominal da economia.")
    st.markdown("---")
    if not df_tesouro.empty:
        st.subheader("Comparativo de Curto Prazo (Últimos 5 Dias)")
        st.plotly_chart(gerar_grafico_ettj_curto_prazo(df_tesouro), use_container_width=True)
        st.markdown("---")
        st.subheader("Comparativo de Longo Prazo (Histórico)")
        st.plotly_chart(gerar_grafico_ettj_longo_prazo(df_tesouro), use_container_width=True)
        st.markdown("---")
        st.subheader("Spread de Juros (10 Anos vs. 2 Anos)")
        st.info("Este gráfico mostra a diferença (spread) entre as taxas dos títulos prefixados com juros semestrais (NTN-Fs) com vencimentos próximos de 10 e 2 anos. Um spread positivo indica uma curva inclinada, o que é típico. Spreads negativos (curva invertida) são raros e podem sinalizar expectativas de recessão.")
        st.plotly_chart(gerar_grafico_spread_juros(df_tesouro), use_container_width=True)
    else:
        st.warning("Não foi possível carregar os dados do Tesouro Direto.")

# --- INÍCIO DA SEÇÃO MODIFICADA ---

elif pagina_selecionada == "Crédito Privado":
    # --- GRÁFICO 1: IDEX-CDI (CÓDIGO ORIGINAL) ---
    st.header("IDEX JGP - Indicador de Crédito Privado (Spread/CDI)")
    st.info(
        "O IDEX-CDI mostra o spread médio (prêmio acima do CDI) exigido pelo mercado para comprar debêntures. "
        "Spreads maiores indicam maior percepção de risco. Filtramos emissores que passaram por eventos de crédito "
        "relevantes (Americanas, Light, etc.) para uma visão mais limpa da tendência."
    )
    df_idex = carregar_dados_idex()
    if not df_idex.empty:
        fig_idex = gerar_grafico_idex(df_idex)
        st.plotly_chart(fig_idex, use_container_width=True)
    else:
        st.warning("Não foi possível carregar os dados do IDEX-CDI para exibição.")

    st.markdown("---")

    # --- GRÁFICO 2: IDEX-INFRA (NOVO GRÁFICO) ---
    st.header("IDEX INFRA - Debêntures de Infraestrutura (Spread/NTN-B)")
    st.info(
        "O IDEX-INFRA mede o spread médio de debêntures incentivadas em relação aos títulos públicos de referência (NTN-Bs). "
        "Ele reflete o prêmio de risco exigido para investir em dívida de projetos de infraestrutura."
    )
    df_idex_infra = carregar_dados_idex_infra()
    if not df_idex_infra.empty:
        fig_idex_infra = gerar_grafico_idex_infra(df_idex_infra)
        st.plotly_chart(fig_idex_infra, use_container_width=True)
    else:
        st.warning("Não foi possível carregar os dados do IDEX INFRA para exibição.")

elif pagina_selecionada == "Econômicos BR":
    st.header("Monitor de Indicadores Econômicos Nacionais")
    st.markdown("---")
    st.subheader("Indicadores Macroeconômicos (BCB)")
    df_bcb, config_bcb = carregar_dados_bcb()
    if not df_bcb.empty:
        data_inicio = st.date_input("Data de Início", df_bcb.index.min().date(), min_value=df_bcb.index.min().date(), max_value=df_bcb.index.max().date(), key='bcb_start')
        data_fim = st.date_input("Data de Fim", df_bcb.index.max().date(), min_value=data_inicio, max_value=df_bcb.index.max().date(), key='bcb_end')
        df_filtrado_bcb = df_bcb.loc[str(data_inicio):str(data_fim)]
        num_cols_bcb = 3
        cols_bcb = st.columns(num_cols_bcb)
        for i, nome_serie in enumerate(df_filtrado_bcb.columns):
            fig_bcb = px.line(df_filtrado_bcb, x=df_filtrado_bcb.index, y=nome_serie, title=nome_serie, template='plotly_dark')
            fig_bcb.update_layout(title_x=0)
            cols_bcb[i % num_cols_bcb].plotly_chart(fig_bcb, use_container_width=True)
    else:
        st.warning("Não foi possível carregar os dados do BCB.")

elif pagina_selecionada == "Commodities":
    st.header("Painel de Preços de Commodities")
    st.markdown("---")
    dados_commodities_categorizados = carregar_dados_commodities()
    if dados_commodities_categorizados:
        st.subheader("Variação Percentual de Preços")
        df_variacao = calcular_variacao_commodities(dados_commodities_categorizados)
        if not df_variacao.empty:
            cols_variacao = [col for col in df_variacao.columns if 'Variação' in col]
            format_dict = {'Preço Atual': '{:,.2f}'}
            format_dict.update({col: '{:+.2%}' for col in cols_variacao})
            st.dataframe(df_variacao.style.format(format_dict, na_rep="-").applymap(colorir_negativo_positivo, subset=cols_variacao), use_container_width=True)
        else:
            st.warning("Não foi possível calcular a variação de preços.")
        st.markdown("---")
        fig_commodities = gerar_dashboard_commodities(dados_commodities_categorizados)
        st.plotly_chart(fig_commodities, use_container_width=True, config={'modeBarButtonsToRemove': ['autoscale']})
    else:
        st.warning("Não foi possível carregar os dados de Commodities.")

elif pagina_selecionada == "Internacional":
    st.header("Monitor de Indicadores Internacionais (FRED)")
    st.markdown("---")
    FRED_API_KEY = 'd78668ca6fc142a1248f7cb9132916b0'
    INDICADORES_FRED = {
        'T10Y2Y': 'Spread da Curva de Juros dos EUA (10 Anos vs 2 Anos)',
        'BAMLH0A0HYM2': 'Spread de Crédito High Yield dos EUA (ICE BofA)',
        'DGS10': 'Juros do Título Americano de 10 Anos (DGS10)'
    }
    df_fred = carregar_dados_fred(FRED_API_KEY, INDICADORES_FRED)
    config_fred = {'modeBarButtonsToRemove': ['autoscale']}
    if not df_fred.empty:
        st.info("O **Spread da Curva de Juros dos EUA (T10Y2Y)** é um dos indicadores mais observados para prever recessões. Quando o valor fica negativo (inversão da curva), historicamente tem sido um sinal de que uma recessão pode ocorrer nos próximos 6 a 18 meses.")
        fig_t10y2y = gerar_grafico_fred(df_fred, 'T10Y2Y', INDICADORES_FRED['T10Y2Y'])
        st.plotly_chart(fig_t10y2y, use_container_width=True, config=config_fred)
        st.markdown("---")
        st.info("O **Spread de Crédito High Yield** mede o prêmio de risco exigido pelo mercado para investir em títulos de empresas com maior risco de crédito. **Spreads crescentes** indicam aversão ao risco (medo) e podem sinalizar uma desaceleração econômica.")
        fig_hy = gerar_grafico_fred(df_fred, 'BAMLH0A0HYM2', INDICADORES_FRED['BAMLH0A0HYM2'])
        st.plotly_chart(fig_hy, use_container_width=True, config=config_fred)
        st.markdown("---")
        st.info("A **taxa de juros do título americano de 10 anos (DGS10)** é uma referência para o custo do crédito global. **Juros em alta** podem indicar expectativas de crescimento econômico e inflação mais fortes.")
        fig_dgs10 = gerar_grafico_fred(df_fred, 'DGS10', INDICADORES_FRED['DGS10'])
        st.plotly_chart(fig_dgs10, use_container_width=True, config=config_fred)
    else:
        st.warning("Não foi possível carregar dados do FRED. Verifique a chave da API ou a conexão com a internet.")

elif pagina_selecionada == "Ações BR":
    st.header("Ferramentas de Análise de Ações Brasileiras")
    st.markdown("---")
    
    # Seção 1: Análise de Ratio (código original mantido)
    st.subheader("Análise de Ratio de Ativos (Long & Short)")
    st.info("Esta ferramenta calcula o ratio entre o preço de dois ativos. "
            "**Interpretação:** Quando o ratio está alto, o Ativo A está caro em relação ao Ativo B. "
            "Quando está baixo, está barato. As bandas mostram desvios padrão que podem indicar pontos de reversão à média.")
    def executar_analise_ratio():
        st.session_state.spinner_placeholder.info(f"Buscando e processando dados para {st.session_state.ticker_a_key} e {st.session_state.ticker_b_key}...")
        close_prices = carregar_dados_acoes([st.session_state.ticker_a_key, st.session_state.ticker_b_key], period="max")
        if close_prices.empty or close_prices.shape[1] < 2:
            st.session_state.spinner_placeholder.error(f"Não foi possível obter dados para ambos os tickers. Verifique os códigos (ex: PETR4.SA) e tente novamente.")
            st.session_state.fig_ratio, st.session_state.kpis_ratio = None, None
        else:
            ratio_analysis = calcular_metricas_ratio(close_prices, st.session_state.ticker_a_key, st.session_state.ticker_b_key, window=st.session_state.window_size_key)
            st.session_state.fig_ratio = gerar_grafico_ratio(ratio_analysis, st.session_state.ticker_a_key, st.session_state.ticker_b_key, window=st.session_state.window_size_key)
            st.session_state.kpis_ratio = calcular_kpis_ratio(ratio_analysis)
            st.session_state.spinner_placeholder.empty()
    col1, col2, col3 = st.columns([0.4, 0.4, 0.2])
    with col1: st.text_input("Ticker do Ativo A (Numerador)", "SMAL11.SA", key="ticker_a_key")
    with col2: st.text_input("Ticker do Ativo B (Denominador)", "BOVA11.SA", key="ticker_b_key")
    with col3: st.number_input("Janela Móvel (dias)", min_value=20, max_value=500, value=252, key="window_size_key")
    st.button("Analisar Ratio", on_click=executar_analise_ratio, use_container_width=True)
    st.session_state.spinner_placeholder = st.empty()
    if 'fig_ratio' not in st.session_state:
        executar_analise_ratio()
    if st.session_state.get('kpis_ratio'):
        kpis = st.session_state.kpis_ratio
        cols = st.columns(5)
        cols[0].metric("Ratio Atual", f"{kpis['atual']:.2f}")
        cols[1].metric("Média Histórica", f"{kpis['media']:.2f}")
        cols[2].metric("Mínimo Histórico", f"{kpis['minimo']:.2f}", f"em {kpis['data_minimo'].strftime('%d/%m/%Y')}")
        cols[3].metric("Máximo Histórico", f"{kpis['maximo']:.2f}", f"em {kpis['data_maximo'].strftime('%d/%m/%Y')}")
        cols[4].metric(label="Variação p/ Média", value=f"{kpis['variacao_para_media']:.2f}%", help="Quanto o Ativo A (numerador) precisa variar para o ratio voltar à média.")
    if st.session_state.get('fig_ratio'):
        st.plotly_chart(st.session_state.fig_ratio, use_container_width=True, config={'modeBarButtonsToRemove': ['autoscale']})
    st.markdown("---")
    
    # Seção 2: Análise de Insiders (código original mantido)
    st.subheader("Radar de Insiders (Movimentações CVM)")
    st.info("Analisa as movimentações de compra e venda de ações feitas por pessoas ligadas à empresa (Controladores, Diretores, etc.), com base nos dados públicos da CVM. Grandes volumes de compra podem indicar confiança na empresa.")
    if st.button("Analisar Movimentações de Insiders do Mês", use_container_width=True):
        with st.spinner("Baixando e processando dados da CVM e YFinance... Isso pode levar alguns minutos."):
            dados_insiders = executar_analise_insiders()
        if dados_insiders:
            df_controladores, df_outros, ultimo_mes = dados_insiders
            st.subheader(f"Dados de {ultimo_mes.strftime('%B de %Y')}")
            if not df_controladores.empty:
                st.write("#### Grupo: Controladores e Vinculados")
                fig_vol_ctrl, fig_rel_ctrl = gerar_graficos_insiders_plotly(df_controladores)
                col1_ctrl, col2_ctrl = st.columns(2)
                with col1_ctrl: st.plotly_chart(fig_vol_ctrl, use_container_width=True)
                with col2_ctrl: st.plotly_chart(fig_rel_ctrl, use_container_width=True)
            else:
                st.warning("Não foram encontrados dados de movimentação para Controladores no último mês.")
            st.markdown("---")
            if not df_outros.empty:
                st.write("#### Grupo: Demais Insiders (Diretores, Conselheiros, etc.)")
                fig_vol_outros, fig_rel_outros = gerar_graficos_insiders_plotly(df_outros)
                col1_outros, col2_outros = st.columns(2)
                with col1_outros: st.plotly_chart(fig_vol_outros, use_container_width=True)
                with col2_outros: st.plotly_chart(fig_rel_outros, use_container_width=True)
            else:
                st.warning("Não foram encontrados dados de movimentação para Demais Insiders no último mês.")
        else:
            st.error("Falha ao processar dados de insiders.")

    st.markdown("---")

