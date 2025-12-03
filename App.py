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
import io  # Adicionado para a nova funcionalidade
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
                line=dict(shape='spline', smoothing=1.0),
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
    """
    Calcula a curva de inflação implícita (breakeven) usando a fotografia mais recente do Tesouro.

    Para cada título prefixado, encontra o IPCA+ com vencimento mais próximo e
    calcula a inflação implícita anualizada:
        (1 + taxa_prefixada) / (1 + juro_real_IPCA) - 1

    Retorna um DataFrame indexado pelo vencimento do prefixado, com colunas:
        - 'Inflação Implícita (% a.a.)'
        - 'Anos até Vencimento'
    """
    if df.empty or 'Data Base' not in df.columns:
        return pd.DataFrame()

    df_recente = df[df['Data Base'] == df['Data Base'].max()].copy()
    if df_recente.empty:
        return pd.DataFrame()

    data_referencia = df_recente['Data Base'].max()
    tipos_ipca = ['Tesouro IPCA+ com Juros Semestrais', 'Tesouro IPCA+']
    df_ipca_raw = df_recente[df_recente['Tipo Titulo'].isin(tipos_ipca)]
    df_prefixados = df_recente[df_recente['Tipo Titulo'] == 'Tesouro Prefixado'].set_index('Data Vencimento')
    df_ipca = df_ipca_raw.sort_values('Tipo Titulo', ascending=False).drop_duplicates('Data Vencimento').set_index('Data Vencimento')
    if df_prefixados.empty or df_ipca.empty: return pd.DataFrame()
    inflacao_implicita = []
    for venc_prefixado, row_prefixado in df_prefixados.iterrows():
        venc_ipca_proximo = min(df_ipca.index, key=lambda d: abs(d - venc_prefixado))
        if abs((venc_ipca_proximo - venc_prefixado).days) < 550:
            taxa_prefixada = row_prefixado['Taxa Compra Manha']
            taxa_ipca = df_ipca.loc[venc_ipca_proximo]['Taxa Compra Manha']

            # Conversão de taxas (% a.a.) para fator e cálculo de breakeven anualizado
            breakeven = (((1 + taxa_prefixada / 100) / (1 + taxa_ipca / 100)) - 1) * 100

            anos_ate_vencimento = (venc_prefixado - data_referencia).days / 365.25
            inflacao_implicita.append({
                'Vencimento do Prefixo': venc_prefixado,
                'Inflação Implícita (% a.a.)': breakeven,
                'Anos até Vencimento': anos_ate_vencimento
            })

    if not inflacao_implicita:
        return pd.DataFrame()

    df_resultado = (
        pd.DataFrame(inflacao_implicita)
        .sort_values('Vencimento do Prefixo')
        .set_index('Vencimento do Prefixo')
    )
    return df_resultado

def gerar_grafico_curva_juros_real_ntnb(df):
    """
    Gera o gráfico da curva de juros real (taxa IPCA+) das NTN-Bs.
    A taxa de juros real é a taxa fixa que as NTN-Bs pagam acima do IPCA.
    """
    if df.empty or 'Data Base' not in df.columns:
        return go.Figure().update_layout(title_text="Não há dados disponíveis.", template='plotly_dark')
    
    # Filtra apenas os títulos NTN-B na data mais recente
    tipos_ntnb = ['Tesouro IPCA+', 'Tesouro IPCA+ com Juros Semestrais']
    df_recente = df[df['Data Base'] == df['Data Base'].max()].copy()
    df_ntnb = df_recente[df_recente['Tipo Titulo'].isin(tipos_ntnb)].copy()
    
    if df_ntnb.empty:
        return go.Figure().update_layout(title_text="Não há dados de NTN-Bs disponíveis.", template='plotly_dark')
    
    # Remove duplicatas, priorizando "com Juros Semestrais" quando houver ambos
    df_ntnb = df_ntnb.sort_values('Tipo Titulo', ascending=False).drop_duplicates('Data Vencimento')
    df_ntnb = df_ntnb.sort_values('Data Vencimento')
    
    # Calcula o prazo até o vencimento em anos
    data_ref = df_recente['Data Base'].max()
    df_ntnb['Anos até Vencimento'] = (
        (pd.to_datetime(df_ntnb['Data Vencimento']) - data_ref).dt.days / 365.25
    )
    
    # Cria o gráfico
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_ntnb['Anos até Vencimento'],
        y=df_ntnb['Taxa Compra Manha'],
        mode='lines',
        line=dict(color='#4CAF50', width=2.5, shape='spline', smoothing=1.0),
        name='Juros Real (IPCA+)',
        hovertemplate=(
            "Vencimento: %{customdata[0]}<br>"
            "Prazo: %{x:.1f} anos<br>"
            "Taxa Real: %{y:.2f}% a.a.<extra></extra>"
        ),
        customdata=np.stack([
            df_ntnb['Data Vencimento'].dt.strftime('%d/%m/%Y')
        ], axis=-1)
    ))
    
    fig.update_layout(
        title=f'Curva de Juros Real (NTN-Bs) - {data_ref.strftime("%d/%m/%Y")}',
        template='plotly_dark',
        title_x=0,
        xaxis_title='Prazo até o Vencimento (anos)',
        yaxis_title='Taxa de Juros Real (% a.a.)',
        showlegend=False
    )
    
    fig.update_yaxes(tickformat=".2f")
    
    return fig

# --- INÍCIO DA FUNÇÃO ATUALIZADA ---

@st.cache_data
def gerar_grafico_spread_juros(df):
    """
    Calcula o spread de juros 10y vs 2y com vencimentos FIXOS.
    Identifica os títulos NTN-F com vencimentos mais próximos de 2 e 10 anos
    na data mais recente e acompanha o spread desses DOIS títulos ao longo do tempo.
    """
    df_ntnf = df[df['Tipo Titulo'] == 'Tesouro Prefixado com Juros Semestrais'].copy()
    if df_ntnf.empty:
        return go.Figure().update_layout(title_text="Não há dados de Tesouro Prefixado com Juros Semestrais.")

    # 1. Encontrar a data mais recente
    if df_ntnf['Data Base'].empty:
         return go.Figure().update_layout(title_text="Não há dados de Data Base para NTN-F.")
    data_recente = df_ntnf['Data Base'].max()
    
    # 2. Isolar os vencimentos disponíveis na data recente
    df_dia_recente = df_ntnf[df_ntnf['Data Base'] == data_recente]
    vencimentos_recentes = df_dia_recente['Data Vencimento'].unique()

    if len(vencimentos_recentes) < 2:
        return go.Figure().update_layout(title_text="Não há vencimentos suficientes na data mais recente para calcular o spread.")

    # 3. Encontrar os vencimentos fixos (curto e longo)
    target_2y = pd.to_datetime(data_recente) + pd.DateOffset(years=2)
    target_10y = pd.to_datetime(data_recente) + pd.DateOffset(years=10)

    venc_curto_fixo = min(vencimentos_recentes, key=lambda d: abs(d - target_2y))
    venc_longo_fixo = min(vencimentos_recentes, key=lambda d: abs(d - target_10y))

    if venc_curto_fixo == venc_longo_fixo:
        return go.Figure().update_layout(title_text="Não foi possível encontrar vencimentos de 2 e 10 anos distintos.")

    # 4. Criar DataFrames para cada vencimento
    df_curto = df_ntnf[df_ntnf['Data Vencimento'] == venc_curto_fixo][['Data Base', 'Taxa Compra Manha']]
    df_curto = df_curto.rename(columns={'Taxa Compra Manha': 'Taxa Curta'}).set_index('Data Base')
    
    df_longo = df_ntnf[df_ntnf['Data Vencimento'] == venc_longo_fixo][['Data Base', 'Taxa Compra Manha']]
    df_longo = df_longo.rename(columns={'Taxa Compra Manha': 'Taxa Longa'}).set_index('Data Base')

    # 5. Mesclar e calcular o spread
    df_merged = pd.merge(df_curto, df_longo, on='Data Base', how='inner')
    df_merged['Spread'] = (df_merged['Taxa Longa'] - df_merged['Taxa Curta']) * 100  # Em basis points

    if df_merged.empty:
        return go.Figure().update_layout(title_text="Não foi possível calcular o spread (sem dados sobrepostos).")

    df_spread_final = df_merged[['Spread']].dropna().sort_index()

    # --- Plotagem e Filtros (lógica mantida da função original) ---
    
    # Título do gráfico atualizado para refletir a nova lógica
    titulo_grafico = (
        f'Spread de Juros (Fixo): '
        f'NTN-F {venc_longo_fixo.strftime("%Y")} vs. NTN-F {venc_curto_fixo.strftime("%Y")}'
    )
    
    df_plot = df_spread_final.reset_index()
    df_plot.columns = ['Data', 'Spread']
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_plot['Data'],
        y=df_plot['Spread'],
        mode='lines',
        fill='tozeroy',
        line=dict(color='#636EFA', shape='spline', smoothing=1.0),
        name='Spread'
    ))
    
    end_date = df_spread_final.index.max()
    start_date_real = df_spread_final.index.min()
    
    fig.update_layout(
        title=titulo_grafico,
        template='plotly_dark',
        title_x=0,
        yaxis_title="Diferença (Basis Points)",
        xaxis_title="Data",
        showlegend=False
    )
    
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1A", step="year", stepmode="backward"),
                dict(count=2, label="2A", step="year", stepmode="backward"),
                dict(count=5, label="5A", step="year", stepmode="backward"),
                dict(step="all", label="Máx")
            ]),
            bgcolor="#333952",
            font=dict(color="white")
        ),
        rangeslider=dict(visible=False),
        type="date"
    )
    
    # Define a visualização inicial padrão para os últimos 5 anos
    start_date_5y_calculada = end_date - pd.DateOffset(years=5)
    start_date_default = max(start_date_5y_calculada, start_date_real)
    fig.update_xaxes(range=[start_date_default, end_date])

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
        # Converte dias úteis para anos para padronização
        df_data['Anos até Vencimento'] = df_data['Dias Uteis'] / 252  # Aproximadamente 252 dias úteis por ano
        line_style = dict(dash='dash', shape='spline', smoothing=1.0) if not legenda.startswith('Hoje') else dict(shape='spline', smoothing=1.0)
        fig.add_trace(go.Scatter(x=df_data['Anos até Vencimento'], y=df_data['Taxa Compra Manha'], mode='lines', name=legenda, line=line_style))
    fig.update_layout(title_text='Curva de Juros (ETTJ) - Curto Prazo (últimos 5 dias)', title_x=0, xaxis_title='Prazo até o Vencimento (anos)', yaxis_title='Taxa (% a.a.)', template='plotly_dark', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
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
        # Converte dias úteis para anos para padronização
        df_data['Anos até Vencimento'] = df_data['Dias Uteis'] / 252  # Aproximadamente 252 dias úteis por ano
        line_style = dict(dash='dash', shape='spline', smoothing=1.0) if not legenda.startswith('Hoje') else dict(shape='spline', smoothing=1.0)
        fig.add_trace(go.Scatter(x=df_data['Anos até Vencimento'], y=df_data['Taxa Compra Manha'], mode='lines', name=legenda, line=line_style))
    fig.update_layout(title_text='Curva de Juros (ETTJ) - Longo Prazo (Comparativo Histórico)', title_x=0, xaxis_title='Prazo até o Vencimento (anos)', yaxis_title='Taxa (% a.a.)', template='plotly_dark', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
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
    
    # Cria o gráfico usando go.Figure para ter controle completo sobre a linha suavizada
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_merged.index,
        y=df_merged['Spread'],
        mode='lines',
        line=dict(color='#636EFA', shape='spline', smoothing=1.0),
        name='Spread',
        hovertemplate='Data: %{x}<br>Spread: %{y:.2f}%<extra></extra>'
    ))
    
    end_date = df_merged.index.max()
    buttons = []
    periods = {'1A': 365, '2A': 730, '5A': 1825, 'Máx': 'max'}
    for label, days in periods.items():
        start_date = df_merged.index.min() if days == 'max' else end_date - timedelta(days=days)
        buttons.append(dict(method='relayout', label=label, args=[{'xaxis.range': [start_date, end_date], 'yaxis.autorange': True}]))
    
    fig.update_layout(
        title='Spread de Juros 10 Anos: NTN-B (Brasil) vs. Treasury (EUA)',
        template='plotly_dark',
        title_x=0,
        yaxis_title="Diferença (Pontos Percentuais)",
        xaxis_title="Data",
        showlegend=False,
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

# --- FUNÇÕES ADICIONAIS PARA ANÁLISE AVANÇADA DE JUROS ---

@st.cache_data
def calcular_volatilidade_curva(df, tipo_titulo='Tesouro Prefixado', janela_dias=30):
    """
    Calcula a volatilidade (desvio padrão móvel) das taxas por prazo até vencimento.
    Indica períodos de maior incerteza no mercado.
    """
    df_titulo = df[df['Tipo Titulo'] == tipo_titulo].copy()
    if df_titulo.empty:
        return pd.DataFrame()
    
    # Calcula anos até vencimento para cada observação
    df_titulo = df_titulo.sort_values(['Data Base', 'Data Vencimento'])
    df_titulo['Anos até Vencimento'] = (
        (pd.to_datetime(df_titulo['Data Vencimento']) - df_titulo['Data Base']).dt.days / 365.25
    )
    
    # Agrupa por prazo (arredondado para 0.5 anos) e calcula volatilidade
    df_titulo['Prazo_arredondado'] = (df_titulo['Anos até Vencimento'] / 0.5).round() * 0.5
    
    # Cria um DataFrame pivot: linhas = datas, colunas = prazos, valores = taxas
    df_pivot = df_titulo.pivot_table(
        index='Data Base',
        columns='Prazo_arredondado',
        values='Taxa Compra Manha',
        aggfunc='mean'
    )
    
    # Calcula desvio padrão móvel para cada prazo
    df_volatilidade = df_pivot.rolling(window=janela_dias, min_periods=5).std()
    
    return df_volatilidade

def gerar_grafico_volatilidade_curva(df_vol, data_ref, janela_dias=30):
    """
    Gera gráfico de heatmap mostrando a volatilidade da curva ao longo do tempo.
    """
    if df_vol.empty:
        return go.Figure().update_layout(title_text="Não há dados de volatilidade disponíveis.", template='plotly_dark')
    
    # Pega apenas os últimos 2 anos para visualização
    cutoff_date = df_vol.index.max() - pd.DateOffset(years=2)
    df_vol_recente = df_vol[df_vol.index >= cutoff_date].copy()
    
    if df_vol_recente.empty:
        return go.Figure().update_layout(title_text="Não há dados suficientes para visualização.", template='plotly_dark')
    
    # Prepara dados para o heatmap
    prazos = sorted([col for col in df_vol_recente.columns if pd.notna(col)])
    
    if not prazos:
        return go.Figure().update_layout(title_text="Não há prazos disponíveis para visualização.", template='plotly_dark')
    
    # Filtra apenas colunas válidas e remove NaN
    df_vol_clean = df_vol_recente[prazos].dropna(axis=1, how='all')
    prazos_validos = [p for p in prazos if p in df_vol_clean.columns]
    
    if not prazos_validos:
        return go.Figure().update_layout(title_text="Não há dados válidos para visualização.", template='plotly_dark')
    
    fig = go.Figure(data=go.Heatmap(
        z=df_vol_clean[prazos_validos].T.values,
        x=df_vol_clean.index,
        y=[f'{p:.1f} anos' for p in prazos_validos],
        colorscale='YlOrRd',
        colorbar=dict(title="Volatilidade<br>(Desvio Padrão)"),
        hovertemplate='Data: %{x|%d/%m/%Y}<br>Prazo: %{y}<br>Volatilidade: %{z:.3f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Volatilidade da Curva de Juros (Últimos 2 Anos) - Janela {janela_dias} dias',
        template='plotly_dark',
        title_x=0,
        xaxis_title='Data',
        yaxis_title='Prazo até Vencimento',
        height=400
    )
    
    return fig

@st.cache_data
def calcular_historico_spread_com_percentis(df):
    """
    Calcula o histórico do spread 10y vs 2y e adiciona percentis históricos.
    Retorna DataFrame com spread e percentis.
    """
    df_ntnf = df[df['Tipo Titulo'] == 'Tesouro Prefixado com Juros Semestrais'].copy()
    if df_ntnf.empty:
        return pd.DataFrame()
    
    data_recente = df_ntnf['Data Base'].max()
    df_dia_recente = df_ntnf[df_ntnf['Data Base'] == data_recente]
    vencimentos_recentes = df_dia_recente['Data Vencimento'].unique()
    
    if len(vencimentos_recentes) < 2:
        return pd.DataFrame()
    
    target_2y = pd.to_datetime(data_recente) + pd.DateOffset(years=2)
    target_10y = pd.to_datetime(data_recente) + pd.DateOffset(years=10)
    
    venc_curto_fixo = min(vencimentos_recentes, key=lambda d: abs(d - target_2y))
    venc_longo_fixo = min(vencimentos_recentes, key=lambda d: abs(d - target_10y))
    
    if venc_curto_fixo == venc_longo_fixo:
        return pd.DataFrame()
    
    df_curto = df_ntnf[df_ntnf['Data Vencimento'] == venc_curto_fixo][['Data Base', 'Taxa Compra Manha']].set_index('Data Base')
    df_longo = df_ntnf[df_ntnf['Data Vencimento'] == venc_longo_fixo][['Data Base', 'Taxa Compra Manha']].set_index('Data Base')
    
    df_merged = pd.merge(df_curto, df_longo, left_index=True, right_index=True, how='inner')
    df_merged['Spread'] = (df_merged['Taxa Compra Manha_y'] - df_merged['Taxa Compra Manha_x']) * 100
    
    df_spread = df_merged[['Spread']].dropna().sort_index()
    
    # Calcula percentis históricos (valores fixos baseados em todo o histórico)
    spread_values = df_spread['Spread'].values
    percentil_5 = np.percentile(spread_values, 5)
    percentil_25 = np.percentile(spread_values, 25)
    percentil_50 = np.percentile(spread_values, 50)
    percentil_75 = np.percentile(spread_values, 75)
    percentil_95 = np.percentile(spread_values, 95)
    
    # Adiciona os percentis como colunas constantes
    df_spread['Percentil_5'] = percentil_5
    df_spread['Percentil_25'] = percentil_25
    df_spread['Percentil_50'] = percentil_50
    df_spread['Percentil_75'] = percentil_75
    df_spread['Percentil_95'] = percentil_95
    
    return df_spread

def gerar_grafico_spread_com_percentis(df_spread):
    """
    Gera gráfico do spread histórico com faixas de percentis.
    """
    if df_spread.empty:
        return go.Figure().update_layout(title_text="Não há dados de spread disponíveis.", template='plotly_dark')
    
    fig = go.Figure()
    
    # Adiciona áreas de percentis
    fig.add_trace(go.Scatter(
        x=df_spread.index,
        y=df_spread['Percentil_95'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=df_spread.index,
        y=df_spread['Percentil_5'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(255, 0, 0, 0.1)',
        name='Faixa 5-95%',
        hovertemplate='Percentil 5-95%<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=df_spread.index,
        y=df_spread['Percentil_75'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=df_spread.index,
        y=df_spread['Percentil_25'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(255, 165, 0, 0.2)',
        name='Faixa 25-75%',
        hovertemplate='Percentil 25-75%<extra></extra>'
    ))
    
    # Linha da mediana
    fig.add_trace(go.Scatter(
        x=df_spread.index,
        y=df_spread['Percentil_50'],
        mode='lines',
        line=dict(color='yellow', width=2, dash='dash'),
        name='Mediana (50%)',
        hovertemplate='Mediana: %{y:.2f} bps<extra></extra>'
    ))
    
    # Linha do spread atual
    fig.add_trace(go.Scatter(
        x=df_spread.index,
        y=df_spread['Spread'],
        mode='lines',
        line=dict(color='#636EFA', width=2.5, shape='spline', smoothing=1.0),
        name='Spread 10y-2y',
        hovertemplate='Data: %{x|%d/%m/%Y}<br>Spread: %{y:.2f} bps<extra></extra>'
    ))
    
    # Linha zero (inversão)
    fig.add_hline(y=0, line_dash="solid", line_color="red", line_width=1, 
                  annotation_text="Inversão", annotation_position="bottom right")
    
    valor_atual = df_spread['Spread'].iloc[-1]
    percentil_atual = stats.percentileofscore(df_spread['Spread'].values, valor_atual)
    
    fig.update_layout(
        title=f'Spread 10y-2y com Percentis Históricos (Atual: {valor_atual:.1f} bps - Percentil {percentil_atual:.1f}%)',
        template='plotly_dark',
        title_x=0,
        yaxis_title="Spread (Basis Points)",
        xaxis_title="Data",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )
    
    end_date = df_spread.index.max()
    start_date_5y = end_date - pd.DateOffset(years=5)
    fig.update_xaxes(range=[start_date_5y, end_date])
    
    return fig

@st.cache_data
def calcular_decomposicao_taxa_nominal(df):
    """
    Calcula a decomposição: Taxa Nominal = Taxa Real + Inflação Implícita + Prêmio de Risco
    """
    data_ref = df['Data Base'].max()
    df_recente = df[df['Data Base'] == data_ref].copy()
    
    # Pega título prefixado e IPCA+ com vencimentos próximos
    df_prefixado = df_recente[df_recente['Tipo Titulo'] == 'Tesouro Prefixado'].copy()
    tipos_ipca = ['Tesouro IPCA+', 'Tesouro IPCA+ com Juros Semestrais']
    df_ipca = df_recente[df_recente['Tipo Titulo'].isin(tipos_ipca)].copy()
    
    if df_prefixado.empty or df_ipca.empty:
        return pd.DataFrame()
    
    # Remove duplicatas de IPCA
    df_ipca = df_ipca.sort_values('Tipo Titulo', ascending=False).drop_duplicates('Data Vencimento')
    
    decomposicao = []
    
    for _, row_pref in df_prefixado.iterrows():
        venc_pref = row_pref['Data Vencimento']
        taxa_nominal = row_pref['Taxa Compra Manha']
        
        # Encontra IPCA+ mais próximo
        venc_ipca_proximo = min(df_ipca['Data Vencimento'], key=lambda d: abs((d - venc_pref).days))
        if abs((venc_ipca_proximo - venc_pref).days) < 550:
            taxa_real = df_ipca[df_ipca['Data Vencimento'] == venc_ipca_proximo]['Taxa Compra Manha'].iloc[0]
            
            # Inflação implícita = (1+nominal)/(1+real) - 1
            inflacao_implicita = (((1 + taxa_nominal/100) / (1 + taxa_real/100)) - 1) * 100
            
            # Prêmio de risco = residual (pode incluir prêmio de liquidez, etc.)
            premio_risco = taxa_nominal - taxa_real - inflacao_implicita
            
            anos_ate_venc = (venc_pref - data_ref).days / 365.25
            
            decomposicao.append({
                'Prazo (anos)': anos_ate_venc,
                'Taxa Nominal': taxa_nominal,
                'Taxa Real': taxa_real,
                'Inflação Implícita': inflacao_implicita,
                'Prêmio de Risco': premio_risco
            })
    
    if not decomposicao:
        return pd.DataFrame()
    
    return pd.DataFrame(decomposicao).sort_values('Prazo (anos)')

def gerar_grafico_decomposicao_taxa(df_decomp, data_ref):
    """
    Gera gráfico de barras empilhadas mostrando a decomposição da taxa nominal.
    """
    if df_decomp.empty:
        return go.Figure().update_layout(title_text="Não há dados para decomposição.", template='plotly_dark')
    
    fig = go.Figure()
    
    prazos = df_decomp['Prazo (anos)'].values
    
    # Adiciona cada componente como barra empilhada
    fig.add_trace(go.Bar(
        x=prazos,
        y=df_decomp['Taxa Real'],
        name='Taxa Real',
        marker_color='#4CAF50',
        hovertemplate='Prazo: %{x:.1f} anos<br>Taxa Real: %{y:.2f}%<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        x=prazos,
        y=df_decomp['Inflação Implícita'],
        name='Inflação Implícita',
        marker_color='#FFB74D',
        hovertemplate='Prazo: %{x:.1f} anos<br>Inflação Implícita: %{y:.2f}%<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        x=prazos,
        y=df_decomp['Prêmio de Risco'],
        name='Prêmio de Risco',
        marker_color='#F44336',
        hovertemplate='Prazo: %{x:.1f} anos<br>Prêmio de Risco: %{y:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Decomposição da Taxa Nominal - {data_ref.strftime("%d/%m/%Y")}',
        template='plotly_dark',
        title_x=0,
        xaxis_title='Prazo até Vencimento (anos)',
        yaxis_title='Taxa (% a.a.)',
        barmode='stack',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )
    
    return fig

def calcular_alerta_inversao_curva(df):
    """
    Calcula o risco de inversão da curva baseado no spread 10y-2y.
    Retorna um dicionário com informações sobre o alerta.
    """
    df_spread = calcular_historico_spread_com_percentis(df)
    
    if df_spread.empty:
        return None
    
    spread_atual = df_spread['Spread'].iloc[-1]
    spread_medio = df_spread['Spread'].mean()
    spread_std = df_spread['Spread'].std()
    
    # Define níveis de alerta
    limite_critico = 50  # bps
    limite_alto = 100  # bps
    
    if spread_atual < -limite_critico:  # Já invertido
        nivel = "CRÍTICO"
        cor = "🔴"
        mensagem = f"Curva INVERTIDA! Spread: {spread_atual:.1f} bps"
    elif spread_atual < limite_critico:  # Muito próximo de inverter
        nivel = "ALTO"
        cor = "🟠"
        mensagem = f"Risco ALTO de inversão. Spread: {spread_atual:.1f} bps (muito próximo de zero)"
    elif spread_atual < limite_alto:  # Atenção
        nivel = "MÉDIO"
        cor = "🟡"
        mensagem = f"Spread reduzido. Spread: {spread_atual:.1f} bps"
    else:  # Normal
        nivel = "BAIXO"
        cor = "🟢"
        mensagem = f"Spread normal. Spread: {spread_atual:.1f} bps"
    
    percentil = stats.percentileofscore(df_spread['Spread'].values, spread_atual)
    
    return {
        'nivel': nivel,
        'cor': cor,
        'mensagem': mensagem,
        'spread_atual': spread_atual,
        'spread_medio': spread_medio,
        'percentil': percentil,
        'tendencia': 'descendo' if spread_atual < spread_medio else 'subindo'
    }

# --- BLOCO 5: LÓGICA DA PÁGINA DE AÇÕES BR ---
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
# App.py

# ... (código existente das outras funções, como gerar_grafico_idex)

# --- BLOCO 7: LÓGICA DO DASHBOARD DE AMPLITUDE ---
@st.cache_data(ttl=3600*8) # Cache de 8 horas
def obter_tickers_cvm_amplitude():
    """Esta função busca a lista de tickers da CVM."""
    st.info("Buscando lista de tickers da CVM... (Cache de 8h)")
    ano = datetime.now().year
    url = f'https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FCA/DADOS/fca_cia_aberta_{ano}.zip'
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            with z.open(f'fca_cia_aberta_valor_mobiliario_{ano}.csv') as f:
                df = pd.read_csv(f, sep=';', encoding='ISO-8859-1', dtype={'Valor_Mobiliario': 'category', 'Mercado': 'category'})
        df_filtrado = df[(df['Valor_Mobiliario'].isin(['Ações Ordinárias', 'Ações Preferenciais'])) & (df['Mercado'] == 'Bolsa')]
        return df_filtrado['Codigo_Negociacao'].dropna().unique().tolist()
    except Exception as e:
        st.error(f"Erro ao obter tickers da CVM: {e}")
        return None

@st.cache_data(ttl=3600*8) # Cache de 8 horas
def obter_precos_historicos_amplitude(tickers, anos_historico=5):
    """Esta função baixa os preços históricos para a análise de amplitude."""
    st.info(f"Baixando dados de preços de {len(tickers)} ativos... (Cache de 8h)")
    tickers_sa = [ticker + ".SA" for ticker in tickers]
    # Alterado para auto_adjust=False para usar 'Adj Close' explicitamente
    dados_completos = yf.download(
        tickers=tickers_sa,
        start=datetime.now() - timedelta(days=anos_historico*365),
        end=datetime.now(),
        auto_adjust=False,
        progress=False,
        group_by='ticker'
    )
    if not dados_completos.empty:
        # Seleciona 'Adj Close' (preferencial) ou 'Close' como fallback
        price_type = 'Adj Close' if 'Adj Close' in dados_completos.columns.get_level_values(1) else 'Close'
        precos = dados_completos.stack(level=0, future_stack=True)[price_type].unstack(level=1)
        return precos.astype('float32')
    return pd.DataFrame()

# ... (restante das funções)

# (No BLOCO 7: LÓGICA DO DASHBOARD DE AMPLITUDE)

@st.cache_data
def calcular_indicadores_amplitude(_precos_fechamento, rsi_periodo=14):
    """Calcula indicadores de amplitude, incluindo Highs/Lows e McClellan."""
    
    # 1. Market Breadth (MM200)
    mma200 = _precos_fechamento.rolling(window=200, min_periods=50).mean()
    acima_da_media = _precos_fechamento > mma200
    percentual_acima_media = (acima_da_media.sum(axis=1) / _precos_fechamento.notna().sum(axis=1)) * 100
    
    # 2. Categorias MM50 vs MM200 (Seu código existente)
    mma50 = _precos_fechamento.rolling(window=50, min_periods=20).mean()
    total_papeis_validos = _precos_fechamento.notna().sum(axis=1)
    
    cat_red = ((_precos_fechamento < mma50) & (_precos_fechamento < mma200)).sum(axis=1) / total_papeis_validos * 100
    cat_yellow = ((_precos_fechamento < mma50) & (_precos_fechamento > mma200)).sum(axis=1) / total_papeis_validos * 100
    cat_green = ((_precos_fechamento > mma50) & (_precos_fechamento > mma200)).sum(axis=1) / total_papeis_validos * 100

    # 3. IFR (usando pandas_ta)
    ifr_individual = _precos_fechamento.apply(
        lambda x: ta.rsi(x, length=rsi_periodo) if len(x.dropna()) >= rsi_periodo else pd.Series(index=x.index, dtype=float),
        axis=0
    )
    total_valido_ifr = ifr_individual.notna().sum(axis=1)
    sobrecompradas = (ifr_individual > 70).sum(axis=1) / total_valido_ifr * 100
    sobrevendidas = (ifr_individual < 30).sum(axis=1) / total_valido_ifr * 100

    # --- NOVO CÓDIGO: Novas Máximas e Mínimas (52 Semanas / 252 Dias) ---
    # Rolling max/min dos últimos 252 dias
    rolling_max = _precos_fechamento.rolling(window=252, min_periods=200).max()
    rolling_min = _precos_fechamento.rolling(window=252, min_periods=200).min()
    
    # Compara o preço ATUAL com o max/min da janela.
    # Nota: O rolling_max inclui o dia atual. Se hoje for nova máxima, preço == rolling_max.
    new_highs = (_precos_fechamento >= rolling_max).sum(axis=1)
    new_lows = (_precos_fechamento <= rolling_min).sum(axis=1)
    net_highs_lows = new_highs - new_lows

    # --- NOVO CÓDIGO: Oscilador McClellan ---
    # 1. Calcular Avanços e Declínios Diários
    diff_precos = _precos_fechamento.diff()
    advances = (diff_precos > 0).sum(axis=1)
    declines = (diff_precos < 0).sum(axis=1)
    net_advances = advances - declines
    
    # 2. Calcular EMAs (10% e 5% constants aprox. equivalem a 19 e 39 dias)
    # A fórmula clássica usa suavização exponencial da diferença líquida (Net Advances)
    ema_19 = net_advances.ewm(span=19, adjust=False).mean()
    ema_39 = net_advances.ewm(span=39, adjust=False).mean()
    mcclellan_osc = ema_19 - ema_39

    df_amplitude = pd.DataFrame({
        'market_breadth': percentual_acima_media.dropna(),
        'IFR_sobrecompradas': sobrecompradas,
        'IFR_sobrevendidas': sobrevendidas,
        'IFR_net': sobrecompradas - sobrevendidas,
        'IFR_media_geral': ifr_individual.mean(axis=1).dropna(),
        'breadth_red': cat_red,
        'breadth_yellow': cat_yellow,
        'breadth_green': cat_green,
        # Novas Colunas
        'new_highs': new_highs,
        'new_lows': new_lows,
        'net_highs_lows': net_highs_lows,
        'mcclellan': mcclellan_osc
    })
    
    return df_amplitude.dropna()
# (No BLOCO 7, substitua a função 'gerar_grafico_amplitude_mm_stacked' por esta)

def gerar_grafico_amplitude_mm_stacked(df_amplitude_plot):
    """
    Gera o gráfico de amplitude de área com SOBREPOSIÇÃO (MM50/200).
    """
    fig = go.Figure()

    # --- Gráfico de Área com Sobreposição ---
    
    fig.add_trace(go.Scatter(
        x=df_amplitude_plot.index, 
        y=df_amplitude_plot['breadth_green'], 
        name='Acima MM50 e MM200', 
        line=dict(color='#4CAF50'),
        fillcolor='rgba(76, 175, 80, 0.4)', # Verde com 40% opacidade
        fill='tozeroy', # <-- ADICIONADO AQUI: Preenche a área até o eixo Y=0
        mode='lines'
    ))
    
    fig.add_trace(go.Scatter(
        x=df_amplitude_plot.index, 
        y=df_amplitude_plot['breadth_yellow'], 
        name='Abaixo MM50, Acima MM200', 
        line=dict(color='#FFC107'),
        fillcolor='rgba(255, 193, 7, 0.4)', # Amarelo/Laranja com 40% opacidade
        fill='tozeroy', # <-- ADICIONADO AQUI
        mode='lines'
    ))
    
    fig.add_trace(go.Scatter(
        x=df_amplitude_plot.index, 
        y=df_amplitude_plot['breadth_red'], 
        name='Abaixo MM50 e MM200', 
        line=dict(color='#F44336'),
        fillcolor='rgba(244, 67, 54, 0.4)', # Vermelho com 40% opacidade
        fill='tozeroy', # <-- ADICIONADO AQUI
        mode='lines'
    ))

    # Atualiza o layout para um gráfico único
    fig.update_layout(
        title_text='Amplitude de Mercado (MM50/200) - Sobreposto',
        title_x=0,
        template='plotly_dark',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title="% Papéis",
        xaxis_title="Data"
    )
    
    # Define o range do eixo Y das barras para 0-100%
    fig.update_yaxes(range=[0, 100])
    
    # Sincroniza o zoom inicial
    if not df_amplitude_plot.empty:
        fig.update_xaxes(range=[df_amplitude_plot.index.min(), df_amplitude_plot.index.max()])

    return fig
def gerar_grafico_net_highs_lows(df_amplitude):
    """Gera gráfico de área para Net New Highs/Lows - versão otimizada."""
    df_plot = df_amplitude[['net_highs_lows', 'new_highs', 'new_lows']].dropna().copy()
    
    if df_plot.empty:
        return go.Figure().update_layout(title_text="Sem dados disponíveis", template='plotly_dark')
    
    fig = go.Figure()
    
    net_values = df_plot['net_highs_lows']
    positive_vals = net_values.where(net_values >= 0, 0)
    negative_vals = net_values.where(net_values < 0, 0)
    
    fig.add_trace(go.Scatter(
        x=df_plot.index,
        y=positive_vals,
        name='Saldo Positivo',
        mode='lines',
        line=dict(color='#4CAF50', width=1),
        fill='tozeroy',
        fillcolor='rgba(76, 175, 80, 0.5)'
    ))
    
    fig.add_trace(go.Scatter(
        x=df_plot.index,
        y=negative_vals,
        name='Saldo Negativo',
        mode='lines',
        line=dict(color='#F44336', width=1),
        fill='tozeroy',
        fillcolor='rgba(244, 67, 54, 0.5)'
    ))
    
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot['new_highs'], 
        name='Novas Máximas', mode='lines', 
        line=dict(color='#81C784', width=1, dash='dot'), visible='legendonly'
    ))
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot['new_lows'], 
        name='Novas Mínimas', mode='lines', 
        line=dict(color='#E57373', width=1, dash='dot'), visible='legendonly'
    ))

    fig.add_hline(y=0, line_dash="solid", line_color="white", line_width=0.5)

    fig.update_layout(
        title_text='Novas Máximas vs. Novas Mínimas (52 Semanas) - Saldo Líquido',
        title_x=0,
        yaxis_title="Saldo de Papéis",
        xaxis_title="Data",
        template='plotly_dark',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1A", step="year", stepmode="backward"),
                dict(count=2, label="2A", step="year", stepmode="backward"),
                dict(step="all", label="Tudo")
            ]),
            bgcolor="#333952",
            font=dict(color="white")
        )
    )
    
    if len(df_plot) > 252:
        end_date = df_plot.index.max()
        start_date = end_date - pd.DateOffset(years=1)
        fig.update_xaxes(range=[start_date, end_date])
    
    return fig

def gerar_grafico_mcclellan(df_amplitude):
    """Gera o gráfico do Oscilador McClellan com filtro de tempo."""
    series_mcclellan = df_amplitude['mcclellan'].dropna()
    
    if series_mcclellan.empty:
        return go.Figure().update_layout(title_text="Sem dados disponíveis", template='plotly_dark')
    
    fig = go.Figure()
    
    positive_vals = series_mcclellan.where(series_mcclellan >= 0, 0)
    negative_vals = series_mcclellan.where(series_mcclellan < 0, 0)
    
    fig.add_trace(go.Scatter(
        x=series_mcclellan.index,
        y=positive_vals,
        name='Positivo',
        mode='lines',
        line=dict(color='#4CAF50', width=1),
        fill='tozeroy',
        fillcolor='rgba(76, 175, 80, 0.4)'
    ))
    
    fig.add_trace(go.Scatter(
        x=series_mcclellan.index,
        y=negative_vals,
        name='Negativo',
        mode='lines',
        line=dict(color='#F44336', width=1),
        fill='tozeroy',
        fillcolor='rgba(244, 67, 54, 0.4)'
    ))
    
    fig.add_hline(y=0, line_dash="solid", line_color="white", line_width=0.5)
    
    fig.update_layout(
        title_text='Oscilador McClellan (Market Breadth Momentum)',
        title_x=0,
        yaxis_title="Oscilador",
        xaxis_title="Data",
        template='plotly_dark',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1A", step="year", stepmode="backward"),
                dict(count=2, label="2A", step="year", stepmode="backward"),
                dict(step="all", label="Tudo")
            ]),
            bgcolor="#333952",
            font=dict(color="white")
        )
    )
    
    if len(series_mcclellan) > 252:
        end_date = series_mcclellan.index.max()
        start_date = end_date - pd.DateOffset(years=1)
        fig.update_xaxes(range=[start_date, end_date])
    
    return fig
    
def analisar_retornos_por_faixa(df_analise, nome_coluna_indicador, passo, min_range, max_range, sufixo=''):
    """Agrupa os dados em faixas e calcula o retorno médio e a taxa de acerto."""
    bins = list(np.arange(min_range, max_range + passo, passo))
    labels = [f'{i} a {i+passo}{sfixo}' for i, sfixo in zip(np.arange(min_range, max_range, passo), [sufixo]*len(bins))]
    df_analise[f'faixa'] = pd.cut(df_analise[nome_coluna_indicador], bins=bins, labels=labels, right=False, include_lowest=True)

    colunas_retorno = [col for col in df_analise.columns if 'retorno_' in col]
    grouped = df_analise.groupby(f'faixa', observed=True)
    media_resultados = grouped[colunas_retorno].mean()
    
    positivos = grouped[colunas_retorno].agg(lambda x: (x > 0).sum())
    totais = grouped[colunas_retorno].count()
    acerto_resultados = (positivos / totais * 100).fillna(0)
    
    return pd.concat([media_resultados, acerto_resultados], axis=1, keys=['Retorno Médio', 'Taxa de Acerto'])

def gerar_grafico_historico_amplitude(series_dados, titulo, valor_atual, media_hist):
    """Gera um gráfico de linha para o histórico do indicador, com botões de período."""
    # Garante que estamos trabalhando com um DataFrame para facilitar os filtros
    df_plot = series_dados.to_frame(name='valor').dropna()
    if df_plot.empty:
        return go.Figure().update_layout(
            title_text=titulo,
            template='plotly_dark',
            title_x=0
        )

    # Gráfico principal
    fig = px.line(df_plot, x=df_plot.index, y='valor', title=titulo, template='plotly_dark')

    # Linhas horizontais de referência
    fig.add_hline(y=media_hist, line_dash="dash", line_color="gray", annotation_text="Média Hist.")
    fig.add_hline(y=valor_atual, line_dash="dot", line_color="yellow", annotation_text=f"Atual: {valor_atual:.2f}")

    # Configuração geral de layout
    fig.update_layout(
        showlegend=False,
        title_x=0,
        yaxis_title="%",
        xaxis_title="Data"
    )

    # Botões de período no eixo X
    end_date = df_plot.index.max()
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1A", step="year", stepmode="backward"),
                dict(count=3, label="3A", step="year", stepmode="backward"),
                dict(count=5, label="5A", step="year", stepmode="backward"),
                dict(step="all", label="Tudo"),
            ]),
            bgcolor="#333952",
            font=dict(color="white")
        ),
        rangeslider=dict(visible=False),
        type="date"
    )

    # Zoom inicial padrão: últimos 5 anos (ou todo o histórico se menor)
    if len(df_plot) > 252 * 5:
        start_date = end_date - pd.DateOffset(years=5)
        fig.update_xaxes(range=[start_date, end_date])

    return fig

def gerar_histograma_amplitude(series_dados, titulo, valor_atual, media_hist, nbins=50):
    """Gera um histograma de uma série de dados com linhas verticais para o valor atual e a média."""
    fig = px.histogram(series_dados, title=titulo, nbins=nbins, template='plotly_dark')
    fig.add_vline(x=media_hist, line_dash="dash", line_color="gray", annotation_text=f"Média: {media_hist:.2f}", annotation_position="top left")
    fig.add_vline(x=valor_atual, line_dash="dot", line_color="yellow", annotation_text=f"Atual: {valor_atual:.2f}", annotation_position="top right")
    fig.update_layout(showlegend=False, title_x=0)
    return fig

def gerar_heatmap_amplitude(tabela_media, faixa_atual, titulo):
    """Gera um heatmap a partir da tabela de análise de faixas."""
    fig = go.Figure(data=go.Heatmap(
        z=tabela_media.values,
        x=[col.replace('retorno_', '') for col in tabela_media.columns],
        y=tabela_media.index,
        hoverongaps=False, colorscale='RdYlGn',
        text=tabela_media.map(lambda x: f'{x:.1f}%').values,
        texttemplate="%{text}"
    ))
    
    faixas_y = list(tabela_media.index)
    if faixa_atual in faixas_y:
        y_pos = faixas_y.index(faixa_atual)
        fig.add_shape(type="rect", xref="paper", yref="y",
                      x0=0, y0=y_pos-0.5, x1=1, y1=y_pos+0.5,
                      line=dict(color="White", width=4))
                      
    fig.update_layout(title=titulo, template='plotly_dark', yaxis_title='Faixa do Indicador', title_x=0)
    return fig

# --- FIM DO BLOCO 7 ---
# --- FIM DO BLOCO 7 ---

# --- BLOCO 8: LÓGICA DO RADAR DE INSIDERS (NOVO) ---
NOME_ARQUIVO_CACHE = "market_caps.csv"
CACHE_VALIDADE_DIAS = 1

@st.cache_data(ttl=3600*8) # Cache de 8 horas
def baixar_e_extrair_zip_cvm(url, nome_csv_interno):
    """Baixa e extrai um CSV de um arquivo ZIP da CVM em memória."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            with z.open(nome_csv_interno) as f:
                return pd.read_csv(f, sep=';', encoding='ISO-8859-1', on_bad_lines='skip')
    except Exception as e:
        st.error(f"Erro ao baixar ou processar dados da CVM de {url}: {e}")
        return None

def obter_market_cap_individual(ticker):
    """Busca o valor de mercado para um único ticker."""
    if pd.isna(ticker) or not isinstance(ticker, str): return ticker, np.nan
    try:
        stock = yf.Ticker(f"{ticker.strip()}.SA")
        market_cap = stock.info.get('marketCap')
        return ticker, market_cap if market_cap else np.nan
    except Exception:
        return ticker, np.nan

def buscar_market_caps_otimizado(df_lookup, force_refresh=False):
    """Busca os valores de mercado usando cache e processamento paralelo."""
    if force_refresh and os.path.exists(NOME_ARQUIVO_CACHE):
        os.remove(NOME_ARQUIVO_CACHE)

    if not force_refresh and os.path.exists(NOME_ARQUIVO_CACHE):
        data_modificacao = datetime.fromtimestamp(os.path.getmtime(NOME_ARQUIVO_CACHE))
        if (datetime.now() - data_modificacao) < timedelta(days=CACHE_VALIDADE_DIAS):
            df_cache = pd.read_csv(NOME_ARQUIVO_CACHE)
            return pd.merge(df_lookup, df_cache, on="Codigo_Negociacao", how="left")

    market_caps = {}
    tickers_para_buscar = df_lookup['Codigo_Negociacao'].dropna().unique().tolist()
    
    with st.spinner(f"Buscando Market Cap para {len(tickers_para_buscar)} tickers... (pode levar alguns minutos)"):
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_ticker = {executor.submit(obter_market_cap_individual, ticker): ticker for ticker in tickers_para_buscar}
            for future in as_completed(future_to_ticker):
                ticker, market_cap = future.result()
                if pd.notna(market_cap):
                    market_caps[ticker] = market_cap

    df_market_caps = pd.DataFrame(list(market_caps.items()), columns=['Codigo_Negociacao', 'MarketCap'])
    df_market_caps.to_csv(NOME_ARQUIVO_CACHE, index=False)
    return pd.merge(df_lookup, df_market_caps, on="Codigo_Negociacao", how="left")

@st.cache_data
def analisar_dados_insiders(_df_mov, _df_cad, meses_selecionados, force_refresh=False):
    """Executa a análise para os meses selecionados."""
    if not meses_selecionados:
        return pd.DataFrame()

    df_periodo = _df_mov[_df_mov['Ano_Mes'].isin(meses_selecionados)].copy()
    if df_periodo.empty:
        st.warning("Não foram encontrados dados para os meses selecionados.")
        return pd.DataFrame()

    df_periodo['Volume_Net'] = np.where(df_periodo['Tipo_Movimentacao'] == 'Compra à vista', df_periodo['Volume'], -df_periodo['Volume'])
    df_net_total = df_periodo.groupby(['CNPJ_Companhia', 'Nome_Companhia'])['Volume_Net'].sum().reset_index()

    cnpjs_unicos = df_net_total[['CNPJ_Companhia']].drop_duplicates()
    df_tickers = _df_cad[['CNPJ_Companhia', 'Codigo_Negociacao']].dropna().drop_duplicates(subset=['CNPJ_Companhia'])
    df_lookup = pd.merge(cnpjs_unicos, df_tickers, on='CNPJ_Companhia', how='left')
    
    df_market_cap_lookup = buscar_market_caps_otimizado(df_lookup, force_refresh=force_refresh)

    df_final = pd.merge(df_net_total, df_market_cap_lookup, on='CNPJ_Companhia', how='left')
    
    market_cap_para_calculo = df_final['MarketCap'].fillna(0)
    df_final['Volume_vs_MarketCap_Pct'] = np.where(
        market_cap_para_calculo > 0,
        (df_final['Volume_Net'] / market_cap_para_calculo) * 100,
        0
    )

    df_tabela = df_final[[
        'Codigo_Negociacao', 'Nome_Companhia', 'Volume_Net', 'MarketCap', 'Volume_vs_MarketCap_Pct'
    ]].rename(columns={
        'Codigo_Negociacao': 'Ticker', 'Nome_Companhia': 'Empresa',
        'Volume_Net': 'Volume Líquido (R$)', 'MarketCap': 'Valor de Mercado (R$)',
        'Volume_vs_MarketCap_Pct': '% do Market Cap'
    })

    df_tabela = df_tabela.dropna(subset=['Ticker'])
    return df_tabela.sort_values(by='Volume Líquido (R$)', ascending=False).reset_index(drop=True)
# --- INÍCIO DAS NOVAS FUNÇÕES (Adicionar no Bloco 8) ---

@st.cache_data
def criar_lookup_ticker_cnpj(_df_cad):
    """
    Cria um dicionário (lookup table) de Ticker -> CNPJ 
    a partir do dataframe de cadastro da CVM.
    """
    df_tickers = _df_cad[['CNPJ_Companhia', 'Codigo_Negociacao']].dropna()
    # Garante tickers únicos, priorizando o primeiro encontrado (se houver duplicatas)
    df_tickers = df_tickers.drop_duplicates(subset=['Codigo_Negociacao'])
    
    # O ticker da CVM é limpo (ex: PETR4, VALE3)
    return pd.Series(df_tickers['CNPJ_Companhia'].values, index=df_tickers['Codigo_Negociacao']).to_dict()

@st.cache_data
def analisar_historico_insider_por_ticker(_df_mov, cnpj_alvo):
    """
    Filtra e agrega o histórico de volume líquido por mês para um único CNPJ.
    Requer que _df_mov já contenha a coluna 'Ano_Mes'.
    """
    if not cnpj_alvo or _df_mov.empty:
        return pd.DataFrame()

    # Filtra movimentações apenas para o CNPJ da empresa alvo
    df_empresa = _df_mov[_df_mov['CNPJ_Companhia'] == cnpj_alvo].copy()
    if df_empresa.empty:
        return pd.DataFrame()

    # Calcula o Volume Líquido (Compra = positivo, Venda = negativo)
    df_empresa['Volume_Net'] = np.where(
        df_empresa['Tipo_Movimentacao'] == 'Compra à vista',
        df_empresa['Volume'],
        -df_empresa['Volume']
    )

    # Agrupa por Ano_Mes (que já foi pré-calculado na UI) e soma o volume líquido
    df_historico = df_empresa.groupby('Ano_Mes')['Volume_Net'].sum().reset_index()

    # Garante que está ordenado por data para o gráfico
    df_historico = df_historico.sort_values(by='Ano_Mes')
    
    # Converte Ano_Mes para um objeto de data real (1º dia do mês) para o gráfico
    df_historico['Data'] = pd.to_datetime(df_historico['Ano_Mes'] + '-01')

    return df_historico[['Data', 'Volume_Net']]

def gerar_grafico_historico_insider(df_historico, ticker):
    """
    Gera um gráfico de barras Plotly para o histórico de volume líquido de insiders.
    """
    if df_historico.empty:
        return go.Figure().update_layout(
            title_text=f"Não há dados de movimentação 'Compra à vista' ou 'Venda à vista' para {ticker}.",
            template="plotly_dark", 
            title_x=0.5
        )

    # Adiciona uma coluna de cor para o gráfico (Verde para Compra, Vermelho para Venda)
    df_historico['Cor'] = np.where(df_historico['Volume_Net'] > 0, '#4CAF50', '#F44336')

    fig = px.bar(
        df_historico,
        x='Data',
        y='Volume_Net',
        title=f'Histórico de Volume Líquido Mensal de Insiders: {ticker.upper()}',
        template='plotly_dark'
    )

    # Aplica as cores customizadas
    fig.update_traces(marker_color=df_historico['Cor'])

    fig.update_layout(
        title_x=0,
        yaxis_title='Volume Líquido (R$)',
        xaxis_title='Mês',
        showlegend=False
    )
    # Formata o eixo Y para Reais (ex: R$ 1.000.000)
    fig.update_yaxes(tickformat="$,.0f") 
    return fig

# --- FIM DO BLOCO 8 ---

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
            "Juros Brasil",
            "Crédito Privado",
            "Amplitude", 
            "Econômicos BR",
            "Commodities",
            "Internacional",
            "Ações BR",
            "Radar de Insiders",
        ],
        # Ícones da https://icons.getbootstrap.com/
        icons=[
            "graph-up-arrow",
            "wallet2",
            "water", 
            "bar-chart-line-fill",
            "box-seam",
            "globe-americas",
            "kanban-fill",
            "person-check-fill",
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

if pagina_selecionada == "Juros Brasil":
    st.header("Dashboard de Juros do Brasil")
    st.info("Esta página consolida a análise de títulos públicos brasileiros: NTN-Bs (juros reais), títulos prefixados (juros nominais), e indicadores derivados.")
    st.markdown("---")

    if not df_tesouro.empty:
        # --- SEÇÃO 1: CURVAS DE JUROS REAL E INFLACAO IMPLICITA ---
        st.subheader("Curvas de Juros Real e Inflação Implícita")
        
        col_curva_real, col_breakeven = st.columns(2)
        
        with col_curva_real:
            st.markdown("#### Curva de Juros Real (NTN-Bs)")
            st.info("Taxa de juros real (IPCA+) que o mercado exige para diferentes prazos. Representa o retorno real esperado acima da inflação.")
            fig_curva_real = gerar_grafico_curva_juros_real_ntnb(df_tesouro)
            st.plotly_chart(fig_curva_real, use_container_width=True)
        
        with col_breakeven:
            st.markdown("#### Inflação Implícita (Breakeven)")
            st.info("Inflação implícita calculada pela diferença entre títulos prefixados e IPCA+ com vencimentos próximos.")
            df_breakeven = calcular_inflacao_implicita(df_tesouro)
            if not df_breakeven.empty:
                # Prepara dados para uma curva mais intuitiva (prazo vs inflação implícita)
                df_breakeven_plot = df_breakeven.reset_index().rename(columns={'Vencimento do Prefixo': 'Vencimento'})

                # Se por algum motivo a coluna não existir (compatibilidade), calcula na hora
                if 'Anos até Vencimento' not in df_breakeven_plot.columns:
                    data_ref = df_tesouro['Data Base'].max()
                    df_breakeven_plot['Anos até Vencimento'] = (
                        (pd.to_datetime(df_breakeven_plot['Vencimento']) - data_ref).dt.days / 365.25
                    )

                data_ref = df_tesouro['Data Base'].max()

                fig_breakeven = go.Figure()
                fig_breakeven.add_trace(go.Scatter(
                    x=df_breakeven_plot['Anos até Vencimento'],
                    y=df_breakeven_plot['Inflação Implícita (% a.a.)'],
                    mode='lines',
                    line=dict(color='#FFB74D', width=2, shape='spline', smoothing=1.0),
                    name='Inflação Implícita',
                    hovertemplate=(
                        "Vencimento: %{customdata[0]}<br>"
                        "Prazo: %{x:.1f} anos<br>"
                        "Inflação Implícita: %{y:.2f}%<extra></extra>"
                    ),
                    customdata=np.stack([
                        df_breakeven_plot['Vencimento'].dt.strftime('%d/%m/%Y')
                    ], axis=-1)
                ))

                fig_breakeven.update_layout(
                    title=f'Curva de Inflação Implícita (Breakeven) - {data_ref.strftime("%d/%m/%Y")}',
                    template='plotly_dark',
                    title_x=0,
                    xaxis_title='Prazo até o Vencimento (anos)',
                    yaxis_title='Inflação Implícita (% a.a.)',
                    showlegend=False
                )

                fig_breakeven.update_yaxes(tickformat=".2f")

                st.plotly_chart(fig_breakeven, use_container_width=True)
            else:
                st.warning("Não há pares de títulos para calcular a inflação implícita hoje.")
        
        st.markdown("---")
        
        # --- SEÇÃO 2: ANÁLISE HISTÓRICA DE NTN-Bs ---
        st.subheader("Análise Histórica de NTN-Bs")
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
                "Analisar por:", ('Taxa', 'PU'),
                horizontal=True, key='metrica_ntnb',
                help="Analisar por Taxa de Compra ou Preço Unitário (PU)"
            )
        coluna_metrica = 'Taxa Compra Manha' if metrica_escolhida == 'Taxa' else 'PU Compra Manha'

        # Gera e exibe o gráfico
        fig_hist_ntnb = gerar_grafico_ntnb_multiplos_vencimentos(
            df_ntnb_all, vencimentos_selecionados, metrica=coluna_metrica
        )
        st.plotly_chart(fig_hist_ntnb, use_container_width=True)

        st.markdown("---")
        
        # --- SEÇÃO 3: ETTJ - CURTO E LONGO PRAZO ---
        st.subheader("Curva de Juros Nominal (ETTJ)")
        
        col_ettj_curto, col_ettj_long = st.columns(2)
        
        with col_ettj_curto:
            st.markdown("#### ETTJ - Curto Prazo")
            st.info("Estrutura a termo da taxa de juros nominal (prefixados) nos últimos 5 dias úteis.")
            st.plotly_chart(gerar_grafico_ettj_curto_prazo(df_tesouro), use_container_width=True)
        
        with col_ettj_long:
            st.markdown("#### ETTJ - Comparativo Histórico (Longo Prazo)")
            st.info("Evolução da curva de juros nominal ao longo do tempo (1 semana, 1 mês, 3 meses, 6 meses, 1 ano atrás).")
            st.plotly_chart(gerar_grafico_ettj_longo_prazo(df_tesouro), use_container_width=True)
        
        st.markdown("---")
        
        # --- SEÇÃO 4: SPREADS ---
        st.subheader("Spreads de Juros")
        
        col_spread_2y10y, col_spread_br_eua = st.columns(2)
        
        with col_spread_2y10y:
            st.markdown("#### Spread de Juros (10 Anos vs. 2 Anos)")
            st.info("Diferença entre as taxas dos títulos prefixados (NTN-Fs) com vencimentos próximos de 10 e 2 anos. Spread positivo = curva inclinada (normal). Spread negativo = curva invertida (sinal de alerta).")
            st.plotly_chart(gerar_grafico_spread_juros(df_tesouro), use_container_width=True, config={'modeBarButtonsToRemove': ['autoscale']})
        
        with col_spread_br_eua:
            st.markdown("#### Spread de Juros: Brasil vs. EUA")
            st.info("Diferença entre a taxa da NTN-B de ~10 anos e o título americano de 10 anos (DGS10). Indica o prêmio de risco país.")
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
        
        st.markdown("---")
        
        # --- SEÇÃO 5: ANÁLISE DE VOLATILIDADE DA CURVA ---
        st.subheader("Análise de Volatilidade da Curva")
        st.info("A volatilidade da curva de juros indica períodos de maior incerteza no mercado. Valores mais altos sugerem maior instabilidade e mudanças frequentes nas expectativas.")
        
        col_volatilidade_info, col_volatilidade_graf = st.columns([1, 2])
        
        with col_volatilidade_info:
            data_ref_vol = df_tesouro['Data Base'].max()
            janela_volatilidade = st.slider("Janela de Volatilidade (dias)", min_value=10, max_value=90, value=30, step=5, key='janela_vol')
            
            df_vol = calcular_volatilidade_curva(df_tesouro, janela_dias=janela_volatilidade)
            if not df_vol.empty:
                # Calcula métricas de volatilidade atual
                volatilidade_atual_media = df_vol.iloc[-1].mean()
                volatilidade_max = df_vol.max().max()
                volatilidade_min = df_vol.min().min()
                
                st.metric("Volatilidade Média Atual", f"{volatilidade_atual_media:.3f}%")
                st.metric("Volatilidade Máxima Histórica", f"{volatilidade_max:.3f}%")
                st.metric("Volatilidade Mínima Histórica", f"{volatilidade_min:.3f}%")
                
                # Identifica prazo com maior volatilidade atual
                volatilidades_atuais = df_vol.iloc[-1].dropna()
                if not volatilidades_atuais.empty:
                    prazo_mais_volatil = volatilidades_atuais.idxmax()
                    st.info(f"**Prazo mais volátil:** {prazo_mais_volatil:.1f} anos ({volatilidades_atuais[prazo_mais_volatil]:.3f}%)")
        
        with col_volatilidade_graf:
            if not df_vol.empty:
                fig_vol = gerar_grafico_volatilidade_curva(df_vol, data_ref_vol, janela_dias=janela_volatilidade)
                st.plotly_chart(fig_vol, use_container_width=True)
            else:
                st.warning("Não foi possível calcular a volatilidade da curva.")
        
        st.markdown("---")
        
        # --- SEÇÃO 6: COMPARATIVO HISTÓRICO DE SPREADS COM PERCENTIS ---
        st.subheader("Análise Histórica de Spread 10y-2y com Percentis")
        st.info("Este gráfico mostra a evolução histórica do spread 10y-2y com faixas de percentis. Permite identificar quando o spread atual está em níveis extremos comparado ao histórico.")
        
        df_spread_percentis = calcular_historico_spread_com_percentis(df_tesouro)
        if not df_spread_percentis.empty:
            fig_spread_perc = gerar_grafico_spread_com_percentis(df_spread_percentis)
            st.plotly_chart(fig_spread_perc, use_container_width=True)
            
            # Métricas adicionais
            spread_atual = df_spread_percentis['Spread'].iloc[-1]
            percentil_atual = stats.percentileofscore(df_spread_percentis['Spread'].values, spread_atual)
            
            col_p5, col_p25, col_p50, col_p75, col_p95 = st.columns(5)
            col_p5.metric("Percentil 5%", f"{df_spread_percentis['Percentil_5'].iloc[-1]:.1f} bps")
            col_p25.metric("Percentil 25%", f"{df_spread_percentis['Percentil_25'].iloc[-1]:.1f} bps")
            col_p50.metric("Mediana (50%)", f"{df_spread_percentis['Percentil_50'].iloc[-1]:.1f} bps")
            col_p75.metric("Percentil 75%", f"{df_spread_percentis['Percentil_75'].iloc[-1]:.1f} bps")
            col_p95.metric("Percentil 95%", f"{df_spread_percentis['Percentil_95'].iloc[-1]:.1f} bps")
            
            st.info(f"**Spread atual:** {spread_atual:.1f} bps | **Percentil histórico:** {percentil_atual:.1f}%")
        else:
            st.warning("Não foi possível calcular o histórico de spreads com percentis.")
        
        st.markdown("---")
        
        # --- SEÇÃO 7: DECOMPOSIÇÃO DA TAXA NOMINAL ---
        st.subheader("Decomposição da Taxa Nominal")
        st.info("A taxa nominal pode ser decomposta em: **Taxa Real** (retorno acima da inflação) + **Inflação Implícita** (expectativa de inflação) + **Prêmio de Risco** (liquidez, risco país, etc.)")
        
        data_ref_decomp = df_tesouro['Data Base'].max()
        df_decomposicao = calcular_decomposicao_taxa_nominal(df_tesouro)
        
        if not df_decomposicao.empty:
            fig_decomp = gerar_grafico_decomposicao_taxa(df_decomposicao, data_ref_decomp)
            st.plotly_chart(fig_decomp, use_container_width=True)
            
            # Tabela com valores detalhados
            st.markdown("#### Valores Detalhados por Prazo")
            df_decomp_display = df_decomposicao.copy()
            df_decomp_display = df_decomp_display.round(2)
            df_decomp_display.columns = ['Prazo (anos)', 'Taxa Nominal (%)', 'Taxa Real (%)', 'Inflação Implícita (%)', 'Prêmio de Risco (%)']
            st.dataframe(df_decomp_display.style.format({
                'Prazo (anos)': '{:.1f}',
                'Taxa Nominal (%)': '{:.2f}',
                'Taxa Real (%)': '{:.2f}',
                'Inflação Implícita (%)': '{:.2f}',
                'Prêmio de Risco (%)': '{:.2f}'
            }), use_container_width=True)
        else:
            st.warning("Não foi possível calcular a decomposição da taxa nominal.")
        
        st.markdown("---")
        
        # --- SEÇÃO 8: INDICADOR DE RISCO DE INVERSÃO ---
        st.subheader("Indicador de Risco de Inversão da Curva")
        
        alerta_inversao = calcular_alerta_inversao_curva(df_tesouro)
        
        if alerta_inversao:
            # Card de alerta visual
            nivel = alerta_inversao['nivel']
            cor = alerta_inversao['cor']
            mensagem = alerta_inversao['mensagem']
            spread_atual = alerta_inversao['spread_atual']
            percentil = alerta_inversao['percentil']
            
            # Determina cor do card baseado no nível
            if nivel == "CRÍTICO":
                st.error(f"### {cor} {nivel}: {mensagem}")
            elif nivel == "ALTO":
                st.warning(f"### {cor} {nivel}: {mensagem}")
            elif nivel == "MÉDIO":
                st.info(f"### {cor} {nivel}: {mensagem}")
            else:
                st.success(f"### {cor} {nivel}: {mensagem}")
            
            # Métricas detalhadas
            col_alerta1, col_alerta2, col_alerta3, col_alerta4 = st.columns(4)
            col_alerta1.metric("Spread Atual", f"{spread_atual:.1f} bps")
            col_alerta2.metric("Spread Médio Histórico", f"{alerta_inversao['spread_medio']:.1f} bps")
            col_alerta3.metric("Percentil Histórico", f"{percentil:.1f}%")
            col_alerta4.metric("Tendência", alerta_inversao['tendencia'].upper())
            
            # Explicação
            if nivel in ["CRÍTICO", "ALTO"]:
                st.markdown("""
                **⚠️ O que significa uma curva invertida?**
                - Quando a curva de juros se inverte (taxa de curto prazo > taxa de longo prazo), 
                  historicamente tem sido um forte sinal de recessão futura.
                - O mercado está sinalizando que espera que as taxas futuras sejam menores que as atuais.
                - No Brasil, inversões da curva têm precedido desacelerações econômicas.
                """)
        else:
            st.warning("Não foi possível calcular o indicador de risco de inversão.")
    
    else:
        st.warning("Não foi possível carregar os dados do Tesouro Direto para exibir esta página.")

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
    # SUBSTITUA a seção "Radar de Insiders" existente por ESTE BLOCO DE CÓDIGO
elif pagina_selecionada == "Amplitude":
    st.header("Análise de Amplitude de Mercado (Market Breadth)")
    st.info(
        "Esta seção analisa a força interna do mercado, avaliando o comportamento de um grande número "
        "de ações em vez de apenas o índice. Indicadores de amplitude podem fornecer sinais "
        "antecipados de mudanças na tendência principal do mercado."
    )
    st.markdown("---")

    # Parâmetros da análise
    ATIVO_ANALISE = 'BOVA11.SA'
    ANOS_HISTORICO = 10
    PERIODOS_RETORNO = {'1 Mês': 21, '3 Meses': 63, '6 Meses': 126, '1 Ano': 252}

    if 'analise_amplitude_executada' not in st.session_state:
        st.session_state.analise_amplitude_executada = False

    if st.button("Executar Análise Completa de Amplitude", use_container_width=True):
        with st.spinner("Realizando análise de amplitude... Este processo pode ser demorado na primeira vez."):
            # 1. Obter dados base
            tickers_cvm = obter_tickers_cvm_amplitude()
            if tickers_cvm:
                precos = obter_precos_historicos_amplitude(tickers_cvm, anos_historico=ANOS_HISTORICO)
                dados_bova11 = yf.download(ATIVO_ANALISE, start=precos.index.min(), end=precos.index.max(), auto_adjust=False, progress=False)

                if not precos.empty and not dados_bova11.empty:
                    st.session_state.df_indicadores = calcular_indicadores_amplitude(precos)
                    
                    if 'Adj Close' in dados_bova11.columns:
                        price_series = dados_bova11[['Adj Close']]
                    else:
                        price_series = dados_bova11[['Close']]
                    price_series.columns = ['price']

                    df_analise_base = price_series
                    for nome_periodo, dias in PERIODOS_RETORNO.items():
                        df_analise_base[f'retorno_{nome_periodo}'] = df_analise_base['price'].pct_change(periods=dias).shift(-dias) * 100
                    
                    st.session_state.df_analise_base = df_analise_base.dropna()
                    st.session_state.analise_amplitude_executada = True
                else:
                    st.error("Não foi possível baixar os dados de preços necessários.")
            else:
                st.error("Não foi possível obter a lista de tickers da CVM.")
    
    if st.session_state.analise_amplitude_executada:
        df_indicadores = st.session_state.df_indicadores
        df_analise_base = st.session_state.df_analise_base
        # --- INÍCIO DO BLOCO DE CÓDIGO ATUALIZADO ---
        st.subheader("Visão Geral da Amplitude (MM50/200)")
        
        # Prepara os dados para o gráfico
        colunas_mm = ['breadth_red', 'breadth_yellow', 'breadth_green']
        df_amplitude_mm_plot = df_indicadores[colunas_mm].dropna()
        
        # Chama a função atualizada (agora só precisa de um argumento)
        fig_stacked = gerar_grafico_amplitude_mm_stacked(df_amplitude_mm_plot)
        st.plotly_chart(fig_stacked, use_container_width=True)
        
        st.markdown("---") # Separa do próximo gráfico
        # --- FIM DO BLOCO DE CÓDIGO ATUALIZADO ---
        
        # --- SEÇÃO 1: MARKET BREADTH (MM200) ---
        st.subheader("Análise de Market Breadth (% de Ações acima da MM200)")
        mb_series = df_indicadores['market_breadth']
        valor_atual_mb = mb_series.iloc[-1]
        media_hist_mb = mb_series.mean()
        df_analise_mb = df_analise_base.join(mb_series).dropna()
        resultados_mb = analisar_retornos_por_faixa(df_analise_mb, 'market_breadth', 10, 0, 100, '%')
        passo_mb = 10
        faixa_atual_valor_mb = int(valor_atual_mb // passo_mb) * passo_mb
        faixa_atual_mb = f'{faixa_atual_valor_mb} a {faixa_atual_valor_mb + passo_mb}%'
        
        col1, col2 = st.columns([1,2])
        with col1:
            st.metric("Valor Atual", f"{valor_atual_mb:.2f}%")
            st.metric("Média Histórica", f"{media_hist_mb:.2f}%")
            z_score_mb = (valor_atual_mb - media_hist_mb) / mb_series.std()
            st.metric("Z-Score (Desvios Padrão)", f"{z_score_mb:.2f}")
            percentil_mb = stats.percentileofscore(mb_series, valor_atual_mb)
            st.metric("Percentil Histórico", f"{percentil_mb:.2f}%")
        with col2:
            st.plotly_chart(gerar_grafico_historico_amplitude(mb_series, "Histórico do Market Breadth (5 Anos)", valor_atual_mb, media_hist_mb), use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(gerar_histograma_amplitude(mb_series, "Distribuição Histórica do Market Breadth", valor_atual_mb, media_hist_mb), use_container_width=True)
        with col2:
            st.plotly_chart(gerar_heatmap_amplitude(resultados_mb['Retorno Médio'], faixa_atual_mb, f"Heatmap de Retorno Médio ({ATIVO_ANALISE})"), use_container_width=True)
        
        st.markdown("---")

        # --- SEÇÃO 2: MÉDIA GERAL DO IFR (SEÇÃO ADICIONADA) ---
        st.subheader("Análise da Média Geral do IFR")
        # Limita a análise da média geral do IFR aos últimos 5 anos
        ifr_media_series = df_indicadores['IFR_media_geral']
        if not ifr_media_series.empty:
            cutoff_ifr = ifr_media_series.index.max() - pd.DateOffset(years=5)
            ifr_media_series = ifr_media_series[ifr_media_series.index >= cutoff_ifr]

        valor_atual_ifr_media = ifr_media_series.iloc[-1]
        media_hist_ifr_media = ifr_media_series.mean()
        df_analise_ifr_media = df_analise_base.join(ifr_media_series).dropna()
        resultados_ifr_media = analisar_retornos_por_faixa(df_analise_ifr_media, 'IFR_media_geral', 5, 0, 100, '')
        passo_ifr_media = 5
        faixa_atual_valor_ifr_media = int(valor_atual_ifr_media // passo_ifr_media) * passo_ifr_media
        faixa_atual_ifr_media = f'{faixa_atual_valor_ifr_media} a {faixa_atual_valor_ifr_media + passo_ifr_media}'

        col1, col2 = st.columns([1,2])
        with col1:
            st.metric("Valor Atual", f"{valor_atual_ifr_media:.2f}")
            st.metric("Média Histórica", f"{media_hist_ifr_media:.2f}")
            z_score_ifr_media = (valor_atual_ifr_media - media_hist_ifr_media) / ifr_media_series.std()
            st.metric("Z-Score (Desvios Padrão)", f"{z_score_ifr_media:.2f}")
            percentil_ifr_media = stats.percentileofscore(ifr_media_series, valor_atual_ifr_media)
            st.metric("Percentil Histórico", f"{percentil_ifr_media:.2f}%")
        with col2:
            st.plotly_chart(gerar_grafico_historico_amplitude(ifr_media_series, "Histórico da Média Geral do IFR (5 Anos)", valor_atual_ifr_media, media_hist_ifr_media), use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(gerar_histograma_amplitude(ifr_media_series, "Distribuição Histórica da Média do IFR", valor_atual_ifr_media, media_hist_ifr_media), use_container_width=True)
        with col2:
            st.plotly_chart(gerar_heatmap_amplitude(resultados_ifr_media['Retorno Médio'], faixa_atual_ifr_media, f"Heatmap de Retorno Médio ({ATIVO_ANALISE}) vs Média IFR"), use_container_width=True)
        
        st.markdown("---")

        # --- SEÇÃO 3: NET IFR ---
        st.subheader("Análise de Net IFR (% Sobrecompradas - % Sobrevendidas)")
        # Limita a análise de Net IFR aos últimos 5 anos
        net_ifr_series = df_indicadores['IFR_net']
        if not net_ifr_series.empty:
            cutoff_net_ifr = net_ifr_series.index.max() - pd.DateOffset(years=5)
            net_ifr_series = net_ifr_series[net_ifr_series.index >= cutoff_net_ifr]

        valor_atual_net_ifr = net_ifr_series.iloc[-1]
        media_hist_net_ifr = net_ifr_series.mean()
        df_analise_net_ifr = df_analise_base.join(net_ifr_series).dropna()
        resultados_net_ifr = analisar_retornos_por_faixa(df_analise_net_ifr, 'IFR_net', 10, -100, 100, '%')
        passo_net_ifr = 10
        faixa_atual_valor_net_ifr = int(valor_atual_net_ifr // passo_net_ifr) * passo_net_ifr
        faixa_atual_net_ifr = f'{faixa_atual_valor_net_ifr} a {faixa_atual_valor_net_ifr + passo_net_ifr}%'
        
        col1, col2 = st.columns([1,2])
        with col1:
            st.metric("Valor Atual", f"{valor_atual_net_ifr:.2f}%")
            st.metric("Média Histórica", f"{media_hist_net_ifr:.2f}%")
            z_score_net_ifr = (valor_atual_net_ifr - media_hist_net_ifr) / net_ifr_series.std()
            st.metric("Z-Score (Desvios Padrão)", f"{z_score_net_ifr:.2f}")
            percentil_net_ifr = stats.percentileofscore(net_ifr_series, valor_atual_net_ifr)
            st.metric("Percentil Histórico", f"{percentil_net_ifr:.2f}%")
        with col2:
            st.plotly_chart(gerar_grafico_historico_amplitude(net_ifr_series, "Histórico do Net IFR (5 Anos)", valor_atual_net_ifr, media_hist_net_ifr), use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(gerar_histograma_amplitude(net_ifr_series, "Distribuição Histórica do Net IFR", valor_atual_net_ifr, media_hist_net_ifr, nbins=100), use_container_width=True)
        with col2:
            st.plotly_chart(gerar_heatmap_amplitude(resultados_net_ifr['Retorno Médio'], faixa_atual_net_ifr, f"Heatmap de Retorno Médio ({ATIVO_ANALISE}) vs Net IFR"), use_container_width=True)
# ... (código anterior da seção Net IFR) ...
        
        # --- SEÇÃO 4: NOVAS MÁXIMAS VS MÍNIMAS (NOVO) ---
        st.subheader("Novas Máximas vs. Novas Mínimas (52 Semanas)")
        st.info("Este indicador mostra o saldo líquido de ações atingindo novas máximas de 52 semanas menos aquelas atingindo novas mínimas. Valores positivos indicam força ampla do mercado.")
        
        # Gera e exibe o gráfico
        fig_nh_nl = gerar_grafico_net_highs_lows(df_indicadores)
        st.plotly_chart(fig_nh_nl, use_container_width=True)
        
        st.markdown("---")

        # --- SEÇÃO 5: OSCILADOR MCCLELLAN (NOVO) ---
        st.subheader("Oscilador McClellan")
        st.info("Indicador de momentum de amplitude baseado no saldo de avanços e declínios. Cruzamentos acima de zero indicam entrada de fluxo comprador generalizado; abaixo de zero, fluxo vendedor. Divergências com o preço são sinais fortes de reversão.")
        
        # Gera e exibe o gráfico
        fig_mcclellan = gerar_grafico_mcclellan(df_indicadores)
        st.plotly_chart(fig_mcclellan, use_container_width=True)

        st.markdown("---")

        # --- SEÇÃO 6: CBOE Brazil ETF Volatility Index (VXEWZCLS) ---
        st.subheader("Volatilidade Implícita Brasil (CBOE Brazil ETF Volatility Index - VXEWZ)")
        st.info(
            "O índice **VXEWZ** mede a volatilidade implícita das opções do ETF EWZ (Brasil) negociado nos EUA, "
            "sendo um termômetro de medo/apetite a risco específico para Brasil. "
            "Níveis elevados indicam maior incerteza e aversão ao risco."
        )

        FRED_API_KEY = 'd78668ca6fc142a1248f7cb9132916b0'
        df_vxewz = carregar_dados_fred(FRED_API_KEY, {'VXEWZCLS': 'CBOE Brazil ETF Volatility Index (VXEWZ)'})

        if not df_vxewz.empty:
            fig_vxewz = gerar_grafico_fred(df_vxewz, 'VXEWZCLS', 'CBOE Brazil ETF Volatility Index (VXEWZ)')
            st.plotly_chart(fig_vxewz, use_container_width=True, config={'modeBarButtonsToRemove': ['autoscale']})
        else:
            st.warning("Não foi possível carregar os dados do índice de volatilidade VXEWZ (VXEWZCLS) a partir do FRED.")

# --- ADICIONE TODO O BLOCO ABAIXO ---
elif pagina_selecionada == "Radar de Insiders":
    st.header("Radar de Movimentação de Insiders (CVM)")
    st.info(
        "Esta ferramenta analisa as movimentações de compra e venda à vista por insiders (controladores, diretores, etc.) "
        "informadas à CVM. Os dados são agregados mensalmente para identificar quais empresas tiveram maior volume líquido "
        "de compras ou vendas."
    )
    st.markdown("---")

    ANO_ATUAL = datetime.now().year
    URL_MOVIMENTACOES = f"https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/VLMO/DADOS/vlmo_cia_aberta_{ANO_ATUAL}.zip"
    URL_CADASTRO = f"https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FCA/DADOS/fca_cia_aberta_{ANO_ATUAL}.zip"
    CSV_MOVIMENTACOES = f"vlmo_cia_aberta_con_{ANO_ATUAL}.csv"
    CSV_CADASTRO = f"fca_cia_aberta_valor_mobiliario_{ANO_ATUAL}.csv"

    # Carrega os dados base com cache
    with st.spinner("Baixando e pré-processando dados da CVM..."):
        df_mov_bruto = baixar_e_extrair_zip_cvm(URL_MOVIMENTACOES, CSV_MOVIMENTACOES)
        df_cad_bruto = baixar_e_extrair_zip_cvm(URL_CADASTRO, CSV_CADASTRO)

    if df_mov_bruto is not None and df_cad_bruto is not None:
        df_mov_bruto['Data_Movimentacao'] = pd.to_datetime(df_mov_bruto['Data_Movimentacao'], errors='coerce')
        df_mov_bruto.dropna(subset=['Data_Movimentacao'], inplace=True)
        df_mov_bruto['Ano_Mes'] = df_mov_bruto['Data_Movimentacao'].dt.strftime('%Y-%m')

        meses_disponiveis = sorted(df_mov_bruto['Ano_Mes'].unique(), reverse=True)

        st.subheader("Configurações da Análise")
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
             meses_selecionados = st.multiselect(
                "Selecione um ou mais meses para analisar",
                options=meses_disponiveis,
                default=[meses_disponiveis[0]] if meses_disponiveis else []
            )
        with col2:
            st.write("") # Espaçador
            st.write("") # Espaçador
            force_refresh = st.checkbox("Forçar Refresh", help="Marque para ignorar o cache de Valor de Mercado e buscar os dados mais recentes online (mais lento).")

        if st.button("Analisar Movimentações", use_container_width=True, type="primary"):
            if meses_selecionados:
                df_resultado = analisar_dados_insiders(df_mov_bruto, df_cad_bruto, meses_selecionados, force_refresh)
                
                st.subheader(f"Resultado da Análise para: {', '.join(meses_selecionados)}")
                
                st.dataframe(df_resultado.style.format({
                    'Volume Líquido (R$)': '{:,.0f}',
                    'Valor de Mercado (R$)': '{:,.0f}',
                    '% do Market Cap': '{:.4f}%'
                }), use_container_width=True)

                # Destaques
                st.markdown("---")
                st.subheader("Destaques da Análise")
                cols_destaques = st.columns(3)
                maior_compra = df_resultado.loc[df_resultado['Volume Líquido (R$)'].idxmax()]
                maior_venda = df_resultado.loc[df_resultado['Volume Líquido (R$)'].idxmin()]
                maior_relevancia = df_resultado.loc[df_resultado['% do Market Cap'].abs().idxmax()]

                cols_destaques[0].metric(
                    label=f"📈 Maior Compra Líquida: {maior_compra['Ticker']}",
                    value=f"R$ {maior_compra['Volume Líquido (R$)']:,.0f}"
                )
                cols_destaques[1].metric(
                    label=f"📉 Maior Venda Líquida: {maior_venda['Ticker']}",
                    value=f"R$ {maior_venda['Volume Líquido (R$)']:,.0f}"
                )
                cols_destaques[2].metric(
                    label=f"📊 Maior Relevância (% Mkt Cap): {maior_relevancia['Ticker']}",
                    value=f"{maior_relevancia['% do Market Cap']:.4f}%",
                    help=f"Volume líquido de R$ {maior_relevancia['Volume Líquido (R$)']:,.0f}"
                )
            else:
                st.warning("Por favor, selecione pelo menos um mês para a análise.")
            # --- (INÍCIO DA NOVA SEÇÃO DE HISTÓRICO POR TICKER) ---
        st.markdown("---")
        st.subheader("Analisar Histórico por Ticker")
        st.info("Digite o código de negociação (ex: PETR4, VALE3) para ver o histórico de volume líquido mensal de insiders.")

        # Cria o lookup Ticker -> CNPJ
        # (Isso é rápido por causa do @st.cache_data na função criar_lookup_ticker_cnpj)
        lookup_ticker_cnpj = criar_lookup_ticker_cnpj(df_cad_bruto)

        ticker_input = st.text_input(
            "Digite o Ticker:", 
            key="insider_ticker_input", 
            placeholder="Ex: PETR4"
        ).upper() # Converte para maiúsculas

        if st.button("Buscar Histórico por Ticker", use_container_width=True):
            if ticker_input:
                # Usa o dicionário para encontrar o CNPJ correspondente ao Ticker
                cnpj_alvo = lookup_ticker_cnpj.get(ticker_input)
                
                if not cnpj_alvo:
                    st.error(f"Ticker '{ticker_input}' não encontrado na base de cadastro da CVM. Verifique o código.")
                else:
                    with st.spinner(f"Analisando histórico para {ticker_input}..."):
                        # Passa o df_mov_bruto (que já tem a coluna 'Ano_Mes' criada)
                        # e o CNPJ encontrado
                        df_historico_ticker = analisar_historico_insider_por_ticker(df_mov_bruto, cnpj_alvo)
                        
                        # Gera e exibe o gráfico
                        fig_historico = gerar_grafico_historico_insider(df_historico_ticker, ticker_input)
                        st.plotly_chart(fig_historico, use_container_width=True)
            else:
                st.warning("Por favor, digite um ticker.")
        
# --- (FIM DA NOVA SEÇÃO) ---

    else:
        st.error("Falha ao carregar os dados base da CVM. A análise não pode continuar.")










