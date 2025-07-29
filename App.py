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
    df_filtrado = df[(df['Tipo Titulo'] == tipo) & (df['Data Vencimento'] == vencimento)].sort_values('Data Base')
    titulo = f'Histórico da Taxa de Compra: {tipo} (Venc. {vencimento.strftime("%d/%m/%Y")})' if metrica == 'Taxa Compra Manha' else f'Histórico do Preço Unitário (PU): {tipo} (Venc. {vencimento.strftime("%d/%m/%Y")})'
    eixo_y = "Taxa de Compra (% a.a.)" if metrica == 'Taxa Compra Manha' else "Preço Unitário (R$)"
    fig = px.line(df_filtrado, x='Data Base', y=metrica, title=titulo, template='plotly_dark')
    fig.update_layout(title_x=0, yaxis_title=eixo_y, xaxis_title="Data")
    return fig

@st.cache_data
def calcular_historico_inflacao_implicita(df):
    df_prefixados = df[df['Tipo Titulo'] == 'Tesouro Prefixado'].copy()
    tipos_ipca = ['Tesouro IPCA+ com Juros Semestrais', 'Tesouro IPCA+']
    df_ipca = df[df['Tipo Titulo'].isin(tipos_ipca)].copy()
    if df_prefixados.empty or df_ipca.empty: return pd.Series(dtype=float)
    historico_breakeven = {}
    for data_base, group in df.groupby('Data Base'):
        prefixados_dia = df_prefixados[df_prefixados['Data Base'] == data_base]
        ipca_dia = df_ipca[df_ipca['Data Base'] == data_base]
        if prefixados_dia.empty or ipca_dia.empty: continue
        try:
            target_5y = data_base + pd.DateOffset(years=5)
            venc_prefixado_5y = min(prefixados_dia['Data Vencimento'], key=lambda d: abs(d - target_5y))
            venc_ipca_5y = min(ipca_dia['Data Vencimento'], key=lambda d: abs(d - target_5y))
            taxa_prefixada = prefixados_dia[prefixados_dia['Data Vencimento'] == venc_prefixado_5y]['Taxa Compra Manha'].iloc[0]
            taxa_ipca = ipca_dia[ipca_dia['Data Vencimento'] == venc_ipca_5y]['Taxa Compra Manha'].iloc[0]
            if pd.notna(taxa_prefixada) and pd.notna(taxa_ipca):
                breakeven = (((1 + taxa_prefixada / 100) / (1 + taxa_ipca / 100)) - 1) * 100
                historico_breakeven[data_base] = breakeven
        except (ValueError, IndexError):
            continue
    return pd.Series(historico_breakeven).sort_index()

def gerar_grafico_historico_inflacao(df_historico):
    if df_historico.empty: return go.Figure().update_layout(title_text="Não há dados para gerar o gráfico.")
    fig = px.line(df_historico, title="Histórico da Inflação Implícita (~5 Anos)", template='plotly_dark')
    end_date = df_historico.index.max()
    buttons = []
    periods = {'1A': 365, '2A': 730, '5A': 1825, 'Máx': 'max'}
    for label, days in periods.items():
        start_date = df_historico.index.min() if days == 'max' else end_date - timedelta(days=days)
        buttons.append(dict(method='relayout', label=label, args=[{'xaxis.range': [start_date, end_date], 'yaxis.autorange': True}]))
    fig.update_layout(title_x=0, yaxis_title="Inflação Implícita (% a.a.)", xaxis_title="Data", updatemenus=[dict(type="buttons", direction="right", showactive=True, x=1, xanchor="right", y=1.05, yanchor="bottom", buttons=buttons)])
    return fig

@st.cache_data
def calcular_inflacao_futura_implicita(df):
    df_recente = df[df['Data Base'] == df['Data Base'].max()].copy()
    df_prefixados = df_recente[df_recente['Tipo Titulo'] == 'Tesouro Prefixado'].copy()
    tipos_ipca = ['Tesouro IPCA+ com Juros Semestrais', 'Tesouro IPCA+']
    df_ipca = df_recente[df_recente['Tipo Titulo'].isin(tipos_ipca)].copy()
    if df_prefixados.empty or df_ipca.empty: return pd.DataFrame()
    def get_rate_for_tenor(df_bonds, tenor_years, data_base):
        target_date = data_base + pd.DateOffset(years=tenor_years)
        vencimento = min(df_bonds['Data Vencimento'], key=lambda d: abs(d - target_date))
        T = (vencimento - data_base).days / 365.25
        taxa = df_bonds[df_bonds['Data Vencimento'] == vencimento]['Taxa Compra Manha'].iloc[0] / 100
        return taxa, T
    data_base = df_recente['Data Base'].max()
    tenors = [2, 5, 10]; rates = {'prefix': {}, 'ipca': {}}
    for tenor in tenors:
        try:
            rates['prefix'][tenor] = get_rate_for_tenor(df_prefixados, tenor, data_base)
            rates['ipca'][tenor] = get_rate_for_tenor(df_ipca, tenor, data_base)
        except (ValueError, IndexError): continue
    forwards = []
    for t1, t2 in [(2, 5), (5, 10)]:
        if t1 in rates['prefix'] and t2 in rates['prefix'] and t1 in rates['ipca'] and t2 in rates['ipca']:
            R_prefix_t1, T_prefix_t1 = rates['prefix'][t1]; R_prefix_t2, T_prefix_t2 = rates['prefix'][t2]
            R_ipca_t1, T_ipca_t1 = rates['ipca'][t1]; R_ipca_t2, T_ipca_t2 = rates['ipca'][t2]
            f_nom = (((1 + R_prefix_t2)**T_prefix_t2) / ((1 + R_prefix_t1)**T_prefix_t1))**(1/(T_prefix_t2 - T_prefix_t1)) - 1
            f_real = (((1 + R_ipca_t2)**T_ipca_t2) / ((1 + R_ipca_t1)**T_ipca_t1))**(1/(T_ipca_t2 - T_ipca_t1)) - 1
            if f_real > -1:
                f_bei = ((1 + f_nom) / (1 + f_real) - 1) * 100
                periodo_label = f"{t2-t1} Anos (a partir de {t1} anos)"
                forwards.append({'Período': periodo_label, 'Inflação Implícita Futura (%)': f_bei})
    return pd.DataFrame(forwards)

@st.cache_data
def gerar_grafico_spread_juros(df):
    df_ntnf = df[df['Tipo Titulo'] == 'Tesouro Prefixado com Juros Semestrais'].copy()
    if df_ntnf.empty: return go.Figure().update_layout(title_text="Não há dados de Tesouro Prefixado com Juros Semestrais.")
    data_recente = df_ntnf['Data Base'].max()
    titulos_disponiveis_hoje = df_ntnf[df_ntnf['Data Base'] == data_recente]
    vencimentos_atuais = sorted(titulos_disponiveis_hoje['Data Vencimento'].unique())
    if len(vencimentos_atuais) < 2: return go.Figure().update_layout(title_text="Menos de duas NTN-Fs disponíveis.")
    target_2y, target_10y = data_recente + pd.DateOffset(years=2), data_recente + pd.DateOffset(years=10)
    venc_curto = min(vencimentos_atuais, key=lambda d: abs(d - target_2y))
    venc_longo = min(vencimentos_atuais, key=lambda d: abs(d - target_10y))
    if venc_curto == venc_longo: return go.Figure().update_layout(title_text="Não foi possível encontrar vértices de 2 e 10 anos distintos.")
    df_curto_hist = df_ntnf[df_ntnf['Data Vencimento'] == venc_curto][['Data Base', 'Taxa Compra Manha']].set_index('Data Base')
    df_longo_hist = df_ntnf[df_ntnf['Data Vencimento'] == venc_longo][['Data Base', 'Taxa Compra Manha']].set_index('Data Base')
    df_spread = pd.merge(df_curto_hist, df_longo_hist, on='Data Base', suffixes=('_curto', '_longo')).dropna()
    if df_spread.empty: return go.Figure().update_layout(title_text=f"Não há histórico comum entre as NTN-Fs.")
    df_spread['Spread'] = (df_spread['Taxa Compra Manha_longo'] - df_spread['Taxa Compra Manha_curto']) * 100
    fig = px.area(df_spread, y='Spread', title=f'Spread de Juros: NTN-F ~10 Anos ({pd.to_datetime(venc_longo).year}) vs ~2 Anos ({pd.to_datetime(venc_curto).year})', template='plotly_dark')
    fig.update_layout(title_x=0, yaxis_title="Diferença (Basis Points)", xaxis_title="Data", showlegend=False)
    return fig

def gerar_grafico_ettj_curto_prazo(df):
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
    SERIES_CONFIG = {'Spread Bancário': {'id': 20783}, 'Inadimplência': {'id': 21082}, 'Crédito/PIB': {'id': 20622}, 'Juros Médio': {'id': 20714}, 'Confiança Consumidor': {'id': 4393}, 'IPCA': {'id': 16122}, 'Atraso 15-90d Total': {'id': 21006}, 'Atraso 15-90d Agro': {'id': 21069}, 'Inadimplência Crédito Rural': {'id': 21146}}
    lista_dfs_sucesso, config_sucesso = [], {}
    for name, config in SERIES_CONFIG.items():
        try:
            df_temp = sgs.get({name: config['id']}, start='2010-01-01'); lista_dfs_sucesso.append(df_temp); config_sucesso[name] = config
        except Exception as e: st.warning(f"Não foi possível carregar o indicador '{name}': {e}")
    if not lista_dfs_sucesso: return pd.DataFrame(), {}
    df_full = pd.concat(lista_dfs_sucesso, axis=1); df_full.ffill(inplace=True); df_full.dropna(inplace=True)
    return df_full, config_sucesso

@st.cache_data
def calcular_previsoes_economicas(df):
    """Calcula previsões simples para próximo mês usando diferentes modelos"""
    if df.empty: return pd.DataFrame()

    previsoes = {}
    df_monthly = df.resample('M').last()  # Converter para mensal se necessário

    for coluna in df.columns:
        serie = df_monthly[coluna].dropna()
        if len(serie) < 12: continue  # Precisa de pelo menos 12 observações

        try:
            # Modelo 1: Média Móvel Simples (3 meses)
            ma_3 = serie.rolling(window=3).mean().iloc[-1]

            # Modelo 2: Média Móvel Ponderada (pesos decrescentes)
            pesos = np.array([0.5, 0.3, 0.2])
            if len(serie) >= 3:
                wma = np.average(serie.iloc[-3:], weights=pesos)
            else:
                wma = serie.iloc[-1]

            # Modelo 3: Tendência Linear Simples (últimos 6 meses)
            if len(serie) >= 6:
                x = np.arange(6)
                y = serie.iloc[-6:].values
                z = np.polyfit(x, y, 1)
                trend_forecast = z[0] * 6 + z[1]  # Próximo ponto da tendência
            else:
                trend_forecast = serie.iloc[-1]

            # Modelo 4: Média da variação percentual (momentum)
            if len(serie) >= 6:
                pct_changes = serie.pct_change().dropna().iloc[-5:]  # Últimas 5 variações
                avg_change = pct_changes.mean()
                momentum_forecast = serie.iloc[-1] * (1 + avg_change)
            else:
                momentum_forecast = serie.iloc[-1]

            # Modelo 5: Sazonalidade simples (mesmo mês do ano anterior)
            if len(serie) >= 12:
                seasonal_forecast = serie.iloc[-12]
            else:
                seasonal_forecast = serie.iloc[-1]

            # Previsão final: média dos modelos (ensemble simples)
            modelos = [ma_3, wma, trend_forecast, momentum_forecast, seasonal_forecast]
            modelos_validos = [m for m in modelos if not np.isnan(m)]

            if modelos_validos:
                previsao_final = np.mean(modelos_validos)
                valor_atual = serie.iloc[-1]
                variacao_prevista = ((previsao_final - valor_atual) / valor_atual) * 100

                previsoes[coluna] = {
                    'Valor Atual': valor_atual,
                    'Previsão Próximo Mês': previsao_final,
                    'Variação Prevista (%)': variacao_prevista,
                    'Média Móvel 3M': ma_3,
                    'Tendência Linear': trend_forecast,
                    'Momentum': momentum_forecast,
                    'Sazonalidade': seasonal_forecast
                }
        except Exception as e:
            continue

    if not previsoes:
        return pd.DataFrame()

    df_previsoes = pd.DataFrame.from_dict(previsoes, orient='index')
    return df_previsoes.round(3)

def gerar_grafico_previsao(df_original, indicador, df_previsoes):
    """Gera gráfico com histórico e previsão para um indicador específico"""
    if indicador not in df_original.columns or indicador not in df_previsoes.index:
        return go.Figure().update_layout(title_text=f"Dados não disponíveis para {indicador}")

    # Dados históricos (últimos 24 meses)
    serie_historica = df_original[indicador].dropna().tail(24)

    # Previsão
    previsao = df_previsoes.loc[indicador, 'Previsão Próximo Mês']

    # Criar próxima data (aproximada)
    ultima_data = serie_historica.index[-1]
    if hasattr(ultima_data, 'to_period'):
        proxima_data = (ultima_data.to_period('M') + 1).to_timestamp()
    else:
        proxima_data = ultima_data + pd.DateOffset(months=1)

    # Criar gráfico
    fig = go.Figure()

    # Linha histórica
    fig.add_trace(go.Scatter(
        x=serie_historica.index,
        y=serie_historica.values,
        mode='lines+markers',
        name='Dados Históricos',
        line=dict(color='#636EFA', width=2)
    ))

    # Ponto de previsão
    fig.add_trace(go.Scatter(
        x=[ultima_data, proxima_data],
        y=[serie_historica.iloc[-1], previsao],
        mode='lines+markers',
        name='Previsão',
        line=dict(color='#FF6B6B', width=3, dash='dash'),
        marker=dict(size=8, symbol='diamond')
    ))

    # Zona de confiança (±10% da previsão como exemplo)
    margem = abs(previsao * 0.1)
    fig.add_trace(go.Scatter(
        x=[proxima_data, proxima_data],
        y=[previsao - margem, previsao + margem],
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(255, 107, 107, 0.2)',
        line=dict(color='rgba(255, 107, 107, 0)'),
        name='Zona de Incerteza (±10%)',
        showlegend=False
    ))

    fig.update_layout(
        title=f'Previsão para {indicador}',
        title_x=0,
        template='plotly_dark',
        xaxis_title='Data',
        yaxis_title='Valor',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig

def colorir_previsao(val):
    """Aplica cores às variações previstas"""
    if pd.isna(val) or val == 0: return ''
    return f"color: {'#4CAF50' if val > 0 else '#F44336'}; font-weight: bold"

# --- BLOCO 3: LÓGICA DO DASHBOARD DE COMMODITIES ---
@st.cache_data(ttl=3600*4)
def carregar_dados_commodities():
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
    if pd.isna(val) or val == 0: return ''
    return f"color: {'#4CAF50' if val > 0 else '#F44336'}"

def gerar_dashboard_commodities(dados_preco_por_categoria):
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

# --- BLOCO 4.5: LÓGICA DE PADRÕES SAZONAIS ---
@st.cache_data
def calcular_padroes_sazonais_commodities(dados_por_categoria):
    """Calcula padrões sazonais para commodities por mês"""
    padroes_sazonais = {}
    
    for categoria, df_categoria in dados_por_categoria.items():
        padroes_categoria = {}
        
        for commodity in df_categoria.columns:
            serie = df_categoria[commodity].dropna()
            if len(serie) < 24:  # Precisa de pelo menos 2 anos de dados
                continue
                
            # Calcular retornos mensais
            serie_mensal = serie.resample('M').last()
            retornos_mensais = serie_mensal.pct_change().dropna()
            
            if len(retornos_mensais) < 12:
                continue
            
            # Agrupar por mês e calcular estatísticas
            retornos_por_mes = retornos_mensais.groupby(retornos_mensais.index.month)
            
            estatisticas_mensais = {}
            for mes in range(1, 13):
                if mes in retornos_por_mes.groups:
                    dados_mes = retornos_por_mes.get_group(mes)
                    estatisticas_mensais[mes] = {
                        'retorno_medio': dados_mes.mean() * 100,
                        'volatilidade': dados_mes.std() * 100,
                        'prob_positivo': (dados_mes > 0).mean() * 100,
                        'num_observacoes': len(dados_mes)
                    }
                else:
                    estatisticas_mensais[mes] = {
                        'retorno_medio': 0,
                        'volatilidade': 0,
                        'prob_positivo': 50,
                        'num_observacoes': 0
                    }
            
            padroes_categoria[commodity] = estatisticas_mensais
        
        if padroes_categoria:
            padroes_sazonais[categoria] = padroes_categoria
    
    return padroes_sazonais

@st.cache_data
def calcular_padroes_sazonais_acoes(tickers_lista):
    """Calcula padrões sazonais para ações brasileiras"""
    padroes_acoes = {}
    
    with st.spinner("Analisando padrões sazonais das ações..."):
        for ticker in tickers_lista:
            try:
                # Baixar dados históricos
                dados = yf.download(ticker, period="max", auto_adjust=True, progress=False)
                if dados.empty:
                    continue
                
                serie = dados['Close'].dropna()
                if len(serie) < 500:  # Precisa de dados suficientes
                    continue
                
                # Calcular retornos mensais
                serie_mensal = serie.resample('M').last()
                retornos_mensais = serie_mensal.pct_change().dropna()
                
                if len(retornos_mensais) < 12:
                    continue
                
                # Agrupar por mês
                retornos_por_mes = retornos_mensais.groupby(retornos_mensais.index.month)
                
                estatisticas_mensais = {}
                for mes in range(1, 13):
                    if mes in retornos_por_mes.groups:
                        dados_mes = retornos_por_mes.get_group(mes)
                        estatisticas_mensais[mes] = {
                            'retorno_medio': dados_mes.mean() * 100,
                            'volatilidade': dados_mes.std() * 100,
                            'prob_positivo': (dados_mes > 0).mean() * 100,
                            'num_observacoes': len(dados_mes)
                        }
                    else:
                        estatisticas_mensais[mes] = {
                            'retorno_medio': 0,
                            'volatilidade': 0,
                            'prob_positivo': 50,
                            'num_observacoes': 0
                        }
                
                padroes_acoes[ticker] = estatisticas_mensais
                
            except Exception as e:
                continue
    
    return padroes_acoes

def gerar_grafico_sazonalidade(padroes_dados, titulo_base, tipo='commodity'):
    """Gera gráfico de padrões sazonais"""
    if not padroes_dados:
        return go.Figure().update_layout(title_text="Nenhum dado disponível para análise sazonal")
    
    # Nomes dos meses
    nomes_meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 
                   'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
    
    fig = go.Figure()
    
    if tipo == 'commodity':
        # Para commodities, mostrar por categoria
        for categoria, commodities in padroes_dados.items():
            # Calcular média da categoria
            retornos_categoria = []
            for mes in range(1, 13):
                retornos_mes = []
                for commodity, dados in commodities.items():
                    if dados[mes]['num_observacoes'] > 0:
                        retornos_mes.append(dados[mes]['retorno_medio'])
                
                if retornos_mes:
                    retornos_categoria.append(np.mean(retornos_mes))
                else:
                    retornos_categoria.append(0)
            
            fig.add_trace(go.Scatter(
                x=nomes_meses,
                y=retornos_categoria,
                mode='lines+markers',
                name=categoria,
                line=dict(width=3),
                marker=dict(size=8)
            ))
    
    else:
        # Para ações individuais
        for ticker, dados in list(padroes_dados.items())[:10]:  # Limitar a 10 ações
            retornos_ticker = [dados[mes]['retorno_medio'] for mes in range(1, 13)]
            
            fig.add_trace(go.Scatter(
                x=nomes_meses,
                y=retornos_ticker,
                mode='lines+markers',
                name=ticker.replace('.SA', ''),
                line=dict(width=2),
                marker=dict(size=6)
            ))
    
    # Adicionar linha de referência no zero
    fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                  annotation_text="Neutro (0%)", annotation_position="bottom right")
    
    fig.update_layout(
        title=f'{titulo_base} - Padrões Sazonais (Retorno Médio Mensal)',
        title_x=0,
        template='plotly_dark',
        xaxis_title='Mês',
        yaxis_title='Retorno Médio (%)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified'
    )
    
    return fig

def gerar_heatmap_sazonalidade(padroes_dados, titulo, limite_itens=15):
    """Gera heatmap de sazonalidade"""
    if not padroes_dados:
        return go.Figure().update_layout(title_text="Nenhum dado disponível")
    
    # Preparar dados para o heatmap
    nomes_meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 
                   'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
    
    # Flatten os dados se for por categoria (commodities)
    items_dados = {}
    if any(isinstance(v, dict) and any(isinstance(vv, dict) for vv in v.values()) for v in padroes_dados.values()):
        # É dados de commodity (categoria -> commodity -> mês)
        for categoria, commodities in padroes_dados.items():
            for commodity, dados_commodity in commodities.items():
                items_dados[f"{commodity} ({categoria})"] = dados_commodity
    else:
        # É dados de ações (ticker -> mês)
        items_dados = padroes_dados
    
    # Limitar número de itens
    items_limitados = dict(list(items_dados.items())[:limite_itens])
    
    # Criar matriz de dados
    matriz_retornos = []
    nomes_items = []
    
    for item, dados in items_limitados.items():
        retornos_item = [dados[mes]['retorno_medio'] for mes in range(1, 13)]
        matriz_retornos.append(retornos_item)
        nomes_items.append(item.replace('.SA', '') if '.SA' in item else item)
    
    if not matriz_retornos:
        return go.Figure().update_layout(title_text="Nenhum dado suficiente para heatmap")
    
    fig = go.Figure(data=go.Heatmap(
        z=matriz_retornos,
        x=nomes_meses,
        y=nomes_items,
        colorscale='RdYlGn',
        colorbar=dict(title="Retorno Médio (%)"),
        hoverongaps=False,
        texttemplate="%{z:.1f}%",
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title=f'{titulo} - Heatmap de Sazonalidade',
        title_x=0,
        template='plotly_dark',
        xaxis_title='Mês',
        yaxis_title='Ativo',
        height=max(400, len(nomes_items) * 25)
    )
    
    return fig

def criar_tabela_ranking_sazonal(padroes_dados, mes_selecionado):
    """Cria tabela com ranking de performance para um mês específico"""
    if not padroes_dados:
        return pd.DataFrame()
    
    ranking_data = []
    
    # Flatten os dados se necessário
    items_dados = {}
    if any(isinstance(v, dict) and any(isinstance(vv, dict) for vv in v.values()) for v in padroes_dados.values()):
        for categoria, commodities in padroes_dados.items():
            for commodity, dados_commodity in commodities.items():
                items_dados[f"{commodity}"] = dados_commodity
    else:
        items_dados = padroes_dados
    
    for item, dados in items_dados.items():
        dados_mes = dados[mes_selecionado]
        if dados_mes['num_observacoes'] > 0:
            ranking_data.append({
                'Ativo': item.replace('.SA', '') if '.SA' in item else item,
                'Retorno Médio (%)': dados_mes['retorno_medio'],
                'Prob. Positivo (%)': dados_mes['prob_positivo'],
                'Volatilidade (%)': dados_mes['volatilidade'],
                'Observações': dados_mes['num_observacoes']
            })
    
    if not ranking_data:
        return pd.DataFrame()
    
    df_ranking = pd.DataFrame(ranking_data)
    df_ranking = df_ranking.sort_values('Retorno Médio (%)', ascending=False)
    
    return df_ranking

# --- BLOCO 5: LÓGICA DA PÁGINA DE AÇÕES BR ---
@st.cache_data(ttl=3600*24)
def executar_analise_insiders():
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

# --- FUNÇÃO ATUALIZADA PARA RETORNAR DOIS GRÁFICOS ---
def gerar_graficos_insiders_plotly(df_dados, top_n=10):
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
    try:
        data = yf.download(tickers, period=period, auto_adjust=True)['Close']
        if isinstance(data, pd.Series): 
            data = data.to_frame(tickers[0])
        return data.dropna()
    except Exception:
        return pd.DataFrame()

@st.cache_data
def calcular_metricas_ratio(data, ticker_a, ticker_b, window=252):
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
    if 'Ratio' not in df_metrics or df_metrics['Ratio'].dropna().empty: return None
    ratio_series = df_metrics['Ratio'].dropna()
    kpis = {"atual": ratio_series.iloc[-1], "media": ratio_series.mean(), "minimo": ratio_series.min(), "data_minimo": ratio_series.idxmin(), "maximo": ratio_series.max(), "data_maximo": ratio_series.idxmax()}
    if kpis["atual"] > 0: kpis["variacao_para_media"] = (kpis["media"] / kpis["atual"] - 1) * 100
    else: kpis["variacao_para_media"] = np.inf
    return kpis

def gerar_grafico_ratio(df_metrics, ticker_a, ticker_b, window):
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

# --- CONSTRUÇÃO DA INTERFACE PRINCIPAL COM ABAS ---
st.title("MOBBT")
st.caption(f"Dados atualizados pela última vez em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Juros BR", "Indicadores Econômicos", "Commodities", "Indicadores Internacionais", "Ações BR", "Padrões Sazonais"])

# --- CONTEÚDO DA ABA 1: JUROS BR ---
with tab1:
    st.header("Análise de Títulos do Tesouro Direto")
    df_tesouro = obter_dados_tesouro()
    if not df_tesouro.empty:
        st.subheader("Estrutura a Termo da Taxa de Juros (ETTJ) - Títulos Prefixados")
        st.plotly_chart(gerar_grafico_ettj_curto_prazo(df_tesouro), use_container_width=True)
        st.plotly_chart(gerar_grafico_ettj_longo_prazo(df_tesouro), use_container_width=True)
        st.markdown("---")
        
        st.subheader("Análises da Curva de Juros")
        col_analise1, col_analise2 = st.columns(2)
        
        with col_analise1:
            st.info("A **Inflação Implícita Histórica** mostra a evolução da expectativa do mercado para a inflação média futura (aqui calculada para o vértice de ~5 anos).")
            df_breakeven_hist = calcular_historico_inflacao_implicita(df_tesouro)
            fig_breakeven_hist = gerar_grafico_historico_inflacao(df_breakeven_hist)
            st.plotly_chart(fig_breakeven_hist, use_container_width=True)

            st.markdown("---")

            st.info("A **Inflação Implícita Futura** mostra a expectativa de inflação para um período que começa no futuro (ex: a inflação média por 5 anos, começando daqui a 5 anos).")
            df_fwd_bei = calcular_inflacao_futura_implicita(df_tesouro)
            if not df_fwd_bei.empty:
                fig_fwd_bei = px.bar(
                    df_fwd_bei, x='Período', y='Inflação Implícita Futura (%)',
                    title='Inflação Implícita na Curva Futura', text_auto='.2f'
                )
                fig_fwd_bei.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
                fig_fwd_bei.update_layout(title_x=0, yaxis_title="Inflação (% a.a.)", xaxis_title="Período Futuro")
                st.plotly_chart(fig_fwd_bei, use_container_width=True)
            else:
                st.warning("Não há pares de títulos disponíveis hoje para calcular a inflação futura.")

        with col_analise2:
            st.info("O **Spread de Juros** mostra a diferença entre as taxas de um título longo e um curto. Positivo indica otimismo; negativo (invertido) pode sinalizar recessão.")
            st.plotly_chart(gerar_grafico_spread_juros(df_tesouro), use_container_width=True)
        
        st.markdown("---")

        st.subheader("Spread de Juros (Risco-País): Brasil 10 Anos vs. EUA 10 Anos")
        st.info("Este gráfico mostra a diferença entre a taxa da NTN-B de ~10 anos e a do título americano de 10 anos. É uma medida da percepção de risco do Brasil. **Spreads crescentes** indicam maior risco percebido, enquanto **spreads caindo** sugerem maior confiança no país.")
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
            st.warning("Não foi possível carregar os dados de juros dos EUA para o comparativo.")
        
        st.markdown("---")

        st.subheader("Análise Histórica de Título Individual")
        col1_hist, col2_hist = st.columns(2)
        with col1_hist:
            tipos_disponiveis = sorted(df_tesouro['Tipo Titulo'].unique())
            tipo_selecionado = st.selectbox("Selecione o Tipo de Título", tipos_disponiveis, key='tipo_tesouro')
        with col2_hist:
            vencimentos_disponiveis = sorted(df_tesouro[df_tesouro['Tipo Titulo'] == tipo_selecionado]['Data Vencimento'].unique())
            vencimento_selecionado = st.selectbox("Selecione a Data de Vencimento", vencimentos_disponiveis, format_func=lambda dt: pd.to_datetime(dt).strftime('%d/%m/%Y'), key='venc_tesouro')
        metrica_escolhida = st.radio("Analisar por:", ('Taxa de Compra', 'Preço Unitário (PU)'), horizontal=True, key='metrica_tesouro', help="**Taxa de Compra:** Rentabilidade anual. **Preço Unitário:** Valor do título (efeito da marcação a mercado).")
        coluna_metrica = 'Taxa Compra Manha' if metrica_escolhida == 'Taxa de Compra' else 'PU Compra Manha'
        if vencimento_selecionado:
            st.plotly_chart(gerar_grafico_historico_tesouro(df_tesouro, tipo_selecionado, pd.to_datetime(vencimento_selecionado), metrica=coluna_metrica), use_container_width=True)
    else: st.warning("Não foi possível carregar os dados do Tesouro.")
# --- CONTEÚDO DA ABA 2: INDICADORES ECONÔMICOS ---
with tab2:
    st.header("Monitor de Indicadores Econômicos do Brasil")
    df_bcb, config_bcb = carregar_dados_bcb()
    if not df_bcb.empty:
        # --- Seção de Previsões ---
        st.subheader("📈 Previsões Econômicas (Próximo Mês)")
        st.info("**Modelos utilizados:** Média móvel, tendência linear, momentum, sazonalidade e média móvel ponderada. "
                "A previsão final é um ensemble (média) destes modelos simples. "
                "⚠️ **Importante:** Estas são previsões estatísticas simples, não constituem recomendações de investimento.")

        df_previsoes = calcular_previsoes_economicas(df_bcb)

        if not df_previsoes.empty:
            # Tabela de previsões
            colunas_principais = ['Valor Atual', 'Previsão Próximo Mês', 'Variação Prevista (%)']
            df_display = df_previsoes[colunas_principais].copy()

            # Formatar para exibição
            format_dict = {
                'Valor Atual': '{:.3f}',
                'Previsão Próximo Mês': '{:.3f}',
                'Variação Prevista (%)': '{:+.2f}%'
            }

            st.dataframe(
                df_display.style.format(format_dict).applymap(
                    colorir_previsao, subset=['Variação Prevista (%)']
                ),
                use_container_width=True
            )

            # Seletor para gráfico detalhado
            st.subheader("Análise Detalhada por Indicador")
            indicador_selecionado = st.selectbox(
                "Selecione um indicador para ver previsão detalhada:",
                options=df_previsoes.index.tolist(),
                key='indicador_previsao'
            )

            if indicador_selecionado:
                col1, col2 = st.columns([2, 1])

                with col1:
                    fig_previsao = gerar_grafico_previsao(df_bcb, indicador_selecionado, df_previsoes)
                    st.plotly_chart(fig_previsao, use_container_width=True)

                with col2:
                    st.write("**Detalhes dos Modelos:**")
                    detalhes = df_previsoes.loc[indicador_selecionado]
                    st.metric("Valor Atual", f"{detalhes['Valor Atual']:.3f}")
                    st.metric("Previsão Final", f"{detalhes['Previsão Próximo Mês']:.3f}", 
                             f"{detalhes['Variação Prevista (%)']:+.2f}%")

                    st.write("**Modelos Individuais:**")
                    st.write(f"• Média Móvel 3M: {detalhes['Média Móvel 3M']:.3f}")
                    st.write(f"• Tendência Linear: {detalhes['Tendência Linear']:.3f}")
                    st.write(f"• Momentum: {detalhes['Momentum']:.3f}")
                    st.write(f"• Sazonalidade: {detalhes['Sazonalidade']:.3f}")
        else:
            st.warning("Não foi possível calcular previsões. Dados insuficientes.")

        st.markdown("---")

        # --- Seção histórica original ---
        st.subheader("📊 Dados Históricos")
        data_inicio = st.date_input("Data de Início", df_bcb.index.min().date(), min_value=df_bcb.index.min().date(), max_value=df_bcb.index.max().date(), key='bcb_start')
        data_fim = st.date_input("Data de Fim", df_bcb.index.max().date(), min_value=data_inicio, max_value=df_bcb.index.max().date(), key='bcb_end')
        df_filtrado_bcb = df_bcb.loc[str(data_inicio):str(data_fim)]

        st.subheader("Gráficos Individuais"); num_cols_bcb = 3; cols_bcb = st.columns(num_cols_bcb)
        for i, nome_serie in enumerate(df_filtrado_bcb.columns):
            fig_bcb = px.line(df_filtrado_bcb, x=df_filtrado_bcb.index, y=nome_serie, title=nome_serie, template='plotly_dark')
            fig_bcb.update_layout(title_x=0)
            cols_bcb[i % num_cols_bcb].plotly_chart(fig_bcb, use_container_width=True)
    else: st.warning("Não foi possível carregar os dados do BCB.")

# --- CONTEÚDO DA ABA 3: COMMODITIES ---
with tab3:
    st.header("Painel de Preços de Commodities")
    dados_commodities_categorizados = carregar_dados_commodities()
    if dados_commodities_categorizados:
        st.subheader("Variação Percentual de Preços")
        df_variacao = calcular_variacao_commodities(dados_commodities_categorizados)
        if not df_variacao.empty:
            cols_variacao = [col for col in df_variacao.columns if 'Variação' in col]
            format_dict = {'Preço Atual': '{:,.2f}'}; format_dict.update({col: '{:+.2%}' for col in cols_variacao})
            st.dataframe(df_variacao.style.format(format_dict, na_rep="-").applymap(colorir_negativo_positivo, subset=cols_variacao), use_container_width=True)
        else: st.warning("Não foi possível calcular a variação de preços.")
        st.markdown("---")
        fig_commodities = gerar_dashboard_commodities(dados_commodities_categorizados)
        st.plotly_chart(fig_commodities, use_container_width=True, config={'modeBarButtonsToRemove': ['autoscale']})
    else: st.warning("Não foi possível carregar os dados de Commodities.")

# --- CONTEÚDO DA ABA 4: INDICADORES INTERNACIONAIS ---
with tab4:
    st.header("Monitor de Indicadores Internacionais (FRED)")
    FRED_API_KEY = 'd78668ca6fc142a1248f7cb9132916b0'
    INDICADORES_FRED = {
        'T10Y2Y': 'Spread da Curva de Juros dos EUA (10 Anos vs 2 Anos)',
        'BAMLH0A0HYM2': 'Spread de Crédito High Yield dos EUA (ICE BofA)',
        'DGS10': 'Juros do Título Americano de 10 Anos (DGS10)'
    }
    df_fred = carregar_dados_fred(FRED_API_KEY, INDICADORES_FRED)
    config_fred = {'modeBarButtonsToRemove': ['autoscale']}

    if not df_fred.empty:
        if 'T10Y2Y' in df_fred.columns:
            st.info("O **Spread da Curva de Juros dos EUA (T10Y2Y)** é um dos indicadores mais observados para prever recessões. Quando o valor fica negativo (inversão da curva), historicamente tem sido um sinal de que uma recessão pode ocorrer nos próximos 6 a 18 meses.")
            fig_t10y2y = gerar_grafico_fred(df_fred, 'T10Y2Y', INDICADORES_FRED['T10Y2Y'])
            st.plotly_chart(fig_t10y2y, use_container_width=True, config=config_fred)
        st.markdown("---")
        if 'BAMLH0A0HYM2' in df_fred.columns:
            st.info("O **Spread de Crédito High Yield** mede o prêmio de risco exigido pelo mercado para investir em títulos de empresas com maior risco de crédito. **Spreads crescentes** indicam aversão ao risco (medo) e podem sinalizar uma desaceleração econômica. **Spreads caindo** indicam apetite por risco (otimismo).")
            fig_hy = gerar_grafico_fred(df_fred, 'BAMLH0A0HYM2', INDICADORES_FRED['BAMLH0A0HYM2'])
            st.plotly_chart(fig_hy, use_container_width=True, config=config_fred)
        st.markdown("---")
        if 'DGS10' in df_fred.columns:
            st.info("A **taxa de juros do título americano de 10 anos (DGS10)** é uma referência para o custo do crédito global. **Juros em alta** podem indicar expectativas de crescimento econômico e inflação mais fortes. **Juros em queda** geralmente sinalizam uma busca por segurança ('flight to safety') ou expectativas de desaceleração.")
            fig_dgs10 = gerar_grafico_fred(df_fred, 'DGS10', INDICADORES_FRED['DGS10'])
            st.plotly_chart(fig_dgs10, use_container_width=True, config=config_fred)
    else:
        st.warning("Não foi possível carregar dados do FRED. Verifique a chave da API ou a conexão com a internet.")

# --- CONTEÚDO DA ABA 5: AÇÕES BR ---
with tab5:
    # --- Seção 1: Análise de Ratio ---
    st.header("Análise de Ratio de Ativos (Long & Short)")
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

    # --- Seção 2: Análise de Insiders (LÓGICA ATUALIZADA) ---
    st.header("Análise de Movimentação de Insiders (CVM)")
    st.info("Analisa as movimentações de compra e venda de ações feitas por pessoas ligadas à empresa (Controladores, Diretores, etc.), com base nos dados públicos da CVM. Grandes volumes de compra podem indicar confiança na empresa.")
    
    if st.button("Analisar Movimentações de Insiders do Mês", use_container_width=True):
        with st.spinner("Baixando e processando dados da CVM e YFinance... Isso pode levar alguns minutos na primeira vez."):
            dados_insiders = executar_analise_insiders()
        
        if dados_insiders:
            df_controladores, df_outros, ultimo_mes = dados_insiders
            st.subheader(f"Dados de {ultimo_mes.strftime('%B de %Y')}")

            # Exibição lado a lado para Controladores
            if not df_controladores.empty:
                st.write("#### Grupo: Controladores e Vinculados")
                fig_vol_ctrl, fig_rel_ctrl = gerar_graficos_insiders_plotly(df_controladores)
                col1_ctrl, col2_ctrl = st.columns(2)
                with col1_ctrl:
                    st.plotly_chart(fig_vol_ctrl, use_container_width=True)
                with col2_ctrl:
                    st.plotly_chart(fig_rel_ctrl, use_container_width=True)
            else:
                st.warning("Não foram encontrados dados de movimentação para Controladores no último mês.")
            
            st.markdown("---")

            # Exibição lado a lado para Demais Insiders
            if not df_outros.empty:
                st.write("#### Grupo: Demais Insiders (Diretores, Conselheiros, etc.)")
                fig_vol_outros, fig_rel_outros = gerar_graficos_insiders_plotly(df_outros)
                col1_outros, col2_outros = st.columns(2)
                with col1_outros:
                    st.plotly_chart(fig_vol_outros, use_container_width=True)
                with col2_outros:
                    st.plotly_chart(fig_rel_outros, use_container_width=True)
            else:
                st.warning("Não foram encontrados dados de movimentação para Demais Insiders no último mês.")
        else:
            st.error("Falha ao processar dados de insiders.")

# --- CONTEÚDO DA ABA 6: PADRÕES SAZONAIS ---
with tab6:
    st.header("📅 Análise de Padrões Sazonais")
    st.info("Esta análise mostra tendências históricas por mês do ano para commodities e ações. "
             "**Interpretação:** Meses com retornos médios positivos historicamente tendem a ser mais favoráveis, "
             "mas lembre-se que performance passada não garante resultados futuros.")
    
    # Seletor de análise
    tipo_analise = st.radio(
        "Escolha o tipo de análise:",
        ["Commodities", "Ações Brasileiras"],
        horizontal=True
    )
    
    if tipo_analise == "Commodities":
        st.subheader("Padrões Sazonais - Commodities")
        
        # Carregar dados de commodities (reutilizar da aba 3)
        dados_commodities_sazonal = carregar_dados_commodities()
        
        if dados_commodities_sazonal:
            # Calcular padrões sazonais
            padroes_commodities = calcular_padroes_sazonais_commodities(dados_commodities_sazonal)
            
            if padroes_commodities:
                # Gráficos de linha por categoria
                st.subheader("Tendências Sazonais por Categoria")
                fig_sazonal_commodities = gerar_grafico_sazonalidade(
                    padroes_commodities, 
                    "Commodities", 
                    tipo='commodity'
                )
                st.plotly_chart(fig_sazonal_commodities, use_container_width=True)
                
                st.markdown("---")
                
                # Heatmap detalhado
                st.subheader("Heatmap de Sazonalidade - Commodities")
                st.info("**Verde** = Meses historicamente favoráveis | **Vermelho** = Meses historicamente desfavoráveis")
                
                fig_heatmap_commodities = gerar_heatmap_sazonalidade(
                    padroes_commodities, 
                    "Commodities",
                    limite_itens=20
                )
                st.plotly_chart(fig_heatmap_commodities, use_container_width=True)
                
                st.markdown("---")
                
                # Ranking por mês específico
                st.subheader("Ranking de Performance por Mês")
                nomes_meses = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho', 
                              'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']
                
                mes_selecionado = st.selectbox(
                    "Selecione um mês para ver o ranking:",
                    options=list(range(1, 13)),
                    format_func=lambda x: nomes_meses[x-1],
                    index=datetime.now().month - 1  # Mês atual como padrão
                )
                
                df_ranking_commodities = criar_tabela_ranking_sazonal(padroes_commodities, mes_selecionado)
                
                if not df_ranking_commodities.empty:
                    # Formatar tabela
                    st.dataframe(
                        df_ranking_commodities.style.format({
                            'Retorno Médio (%)': '{:+.2f}%',
                            'Prob. Positivo (%)': '{:.1f}%',
                            'Volatilidade (%)': '{:.2f}%'
                        }).applymap(
                            lambda x: 'color: #4CAF50' if isinstance(x, (int, float)) and x > 0 
                            else 'color: #F44336' if isinstance(x, (int, float)) and x < 0 
                            else '', 
                            subset=['Retorno Médio (%)']
                        ),
                        use_container_width=True
                    )
                else:
                    st.warning("Nenhum dado disponível para o mês selecionado.")
            else:
                st.warning("Não foi possível calcular padrões sazonais para commodities.")
        else:
            st.warning("Dados de commodities não disponíveis.")
    
    else:  # Ações Brasileiras
        st.subheader("Padrões Sazonais - Ações Brasileiras")
        
        # Lista de ações populares para análise
        acoes_populares = [
            'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'ABEV3.SA',
            'WEGE3.SA', 'MGLU3.SA', 'LREN3.SA', 'JBSS3.SA', 'BEEF3.SA',
            'BOVA11.SA', 'SMAL11.SA', 'IVVB11.SA', 'DIVO11.SA', 'XBOV11.SA'
        ]
        
        # Permitir personalização da lista
        with st.expander("Configurar Lista de Ações"):
            acoes_customizadas = st.text_area(
                "Adicione tickers personalizados (um por linha, formato: PETR4.SA):",
                value="\n".join(acoes_populares),
                height=200
            )
            acoes_lista_final = [ticker.strip() for ticker in acoes_customizadas.split('\n') if ticker.strip()]
        
        if st.button("Analisar Padrões Sazonais das Ações", use_container_width=True):
            # Calcular padrões sazonais para ações
            padroes_acoes = calcular_padroes_sazonais_acoes(acoes_lista_final)
            
            if padroes_acoes:
                # Armazenar no session_state para não recalcular
                st.session_state.padroes_acoes_calculados = padroes_acoes
                
                # Gráfico de linha
                st.subheader("Tendências Sazonais - Principais Ações")
                fig_sazonal_acoes = gerar_grafico_sazonalidade(
                    padroes_acoes, 
                    "Ações Brasileiras", 
                    tipo='acao'
                )
                st.plotly_chart(fig_sazonal_acoes, use_container_width=True)
                
                st.markdown("---")
                
                # Heatmap
                st.subheader("Heatmap de Sazonalidade - Ações")
                fig_heatmap_acoes = gerar_heatmap_sazonalidade(
                    padroes_acoes, 
                    "Ações Brasileiras",
                    limite_itens=15
                )
                st.plotly_chart(fig_heatmap_acoes, use_container_width=True)
                
                st.markdown("---")
                
                # Ranking por mês
                st.subheader("Ranking de Performance por Mês - Ações")
                nomes_meses = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho', 
                              'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']
                
                mes_selecionado_acoes = st.selectbox(
                    "Selecione um mês para ver o ranking:",
                    options=list(range(1, 13)),
                    format_func=lambda x: nomes_meses[x-1],
                    index=datetime.now().month - 1,
                    key='mes_acoes'
                )
                
                df_ranking_acoes = criar_tabela_ranking_sazonal(padroes_acoes, mes_selecionado_acoes)
                
                if not df_ranking_acoes.empty:
                    st.dataframe(
                        df_ranking_acoes.style.format({
                            'Retorno Médio (%)': '{:+.2f}%',
                            'Prob. Positivo (%)': '{:.1f}%',
                            'Volatilidade (%)': '{:.2f}%'
                        }).applymap(
                            lambda x: 'color: #4CAF50' if isinstance(x, (int, float)) and x > 0 
                            else 'color: #F44336' if isinstance(x, (int, float)) and x < 0 
                            else '', 
                            subset=['Retorno Médio (%)']
                        ),
                        use_container_width=True
                    )
                else:
                    st.warning("Nenhum dado disponível para o mês selecionado.")
            else:
                st.warning("Não foi possível calcular padrões sazonais para as ações selecionadas.")
        
        # Mostrar resultados salvos se existirem
        elif 'padroes_acoes_calculados' in st.session_state:
            padroes_acoes = st.session_state.padroes_acoes_calculados
            
            st.subheader("Tendências Sazonais - Principais Ações")
            fig_sazonal_acoes = gerar_grafico_sazonalidade(
                padroes_acoes, 
                "Ações Brasileiras", 
                tipo='acao'
            )
            st.plotly_chart(fig_sazonal_acoes, use_container_width=True)
            
            st.markdown("---")
            
            st.subheader("Heatmap de Sazonalidade - Ações")
            fig_heatmap_acoes = gerar_heatmap_sazonalidade(
                padroes_acoes, 
                "Ações Brasileiras",
                limite_itens=15
            )
            st.plotly_chart(fig_heatmap_acoes, use_container_width=True)
    
    st.markdown("---")
    st.info("💡 **Dica de Uso:** Os padrões sazonais podem ajudar a identificar meses historicamente favoráveis para diferentes ativos, "
            "mas sempre considere o contexto econômico atual e outros fatores fundamentais antes de tomar decisões de investimento.")
