# app.py (Versão com Preços Atuais e Variação Recalculada)

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

# --- CONFIGURAÇÃO GERAL DA PÁGINA ---
st.set_page_config(layout="wide", page_title="MOBBT")

# --- BLOCO 1: LÓGICA DO DASHBOARD DO TESOURO DIRETO (COM CACHE EM ARQUIVO) ---
# (Esta seção permanece inalterada)
def obter_dados_tesouro(cache_file='tesouro_data.parquet', max_age_hours=4):
    url = 'https://www.tesourotransparente.gov.br/ckan/dataset/df56aa42-484a-4a59-8184-7676580c81e3/resource/796d2059-14e9-44e3-80c9-2d9e30b405c1/download/precotaxatesourodireto.csv'
    if os.path.exists(cache_file):
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if (datetime.now() - file_mod_time) < timedelta(hours=max_age_hours):
            return pd.read_parquet(cache_file)
    with st.spinner("Cache do Tesouro expirado. Baixando e processando dados atualizados..."):
        try:
            df = pd.read_csv(url, sep=';', decimal=',')
            df['Data Vencimento'] = pd.to_datetime(df['Data Vencimento'], format='%d/%m/%Y')
            df['Data Base'] = pd.to_datetime(df['Data Base'], format='%d/%m/%Y')
            df['Tipo Titulo'] = df['Tipo Titulo'].astype('category')
            df.to_parquet(cache_file)
            return df
        except Exception as e:
            st.error(f"Erro ao baixar dados do Tesouro: {e}")
            if os.path.exists(cache_file):
                return pd.read_parquet(cache_file)
            return pd.DataFrame()

def gerar_grafico_historico_tesouro(df, tipo, vencimento):
    df_filtrado = df[(df['Tipo Titulo'] == tipo) & (df['Data Vencimento'] == vencimento)].sort_values('Data Base')
    fig = px.line(df_filtrado, x='Data Base', y='Taxa Compra Manha',
                  title=f'Histórico da Taxa: {tipo} (Venc. {vencimento.strftime("%d/%m/%Y")})',
                  template='plotly_dark')
    fig.update_layout(title_x=0.5, yaxis_title="Taxa de Compra", xaxis_title="Data")
    return fig

def gerar_grafico_ettj_curto_prazo(df):
    df_prefixado = df[df['Tipo Titulo'] == 'Tesouro Prefixado'].copy()
    if df_prefixado.empty: return go.Figure().update_layout(title_text="Não há dados para 'Tesouro Prefixado'.")
    datas_disponiveis = sorted(df_prefixado['Data Base'].unique())
    data_recente = datas_disponiveis[-1]
    targets = {f'Hoje ({data_recente.strftime("%d/%m/%Y")})': data_recente,
               '1 dia Atrás': data_recente - pd.DateOffset(days=1),'2 dias Atrás': data_recente - pd.DateOffset(days=2),
               '3 dias Atrás': data_recente - pd.DateOffset(days=3),'4 dias Atrás': data_recente - pd.DateOffset(days=4),
               '5 dias Atrás': data_recente - pd.DateOffset(days=5)}
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
    fig.update_layout(title_text='Curva de Juros (ETTJ) - Curto Prazo (últimos 5 dias)', title_x=0.5,
                      xaxis_title='Dias Úteis até o Vencimento', yaxis_title='Taxa (% a.a.)', template='plotly_dark', 
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def gerar_grafico_ettj_longo_prazo(df):
    df_prefixado = df[df['Tipo Titulo'] == 'Tesouro Prefixado'].copy()
    if df_prefixado.empty: return go.Figure().update_layout(title_text="Não há dados para 'Tesouro Prefixado'.")
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
                 datas_para_plotar[legenda_final] = data_real
    fig = go.Figure()
    for legenda, data_base in datas_para_plotar.items():
        df_data = df_prefixado[df_prefixado['Data Base'] == data_base].sort_values('Data Vencimento')
        df_data['Dias Uteis'] = np.busday_count(df_data['Data Base'].values.astype('M8[D]'), df_data['Data Vencimento'].values.astype('M8[D]'))
        line_style = dict(dash='dash') if not legenda.startswith('Hoje') else {}
        fig.add_trace(go.Scatter(x=df_data['Dias Uteis'], y=df_data['Taxa Compra Manha'], mode='lines+markers', name=legenda, line=line_style))
    fig.update_layout(title_text='Curva de Juros (ETTJ) - Longo Prazo (Comparativo Histórico)', title_x=0.5,
                      xaxis_title='Dias Úteis até o Vencimento', yaxis_title='Taxa (% a.a.)', template='plotly_dark',
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

# --- BLOCO 2: LÓGICA DO DASHBOARD DE INDICADORES ECONÔMICOS (EXPANDIDO E ROBUSTO) ---
# (Esta seção permanece inalterada)
@st.cache_data(ttl=3600*4)
def carregar_dados_bcb():
    SERIES_CONFIG = {'Spread Bancário': {'id': 20783}, 'Inadimplência': {'id': 21082}, 'Crédito/PIB': {'id': 20622}, 
                     'Juros Médio': {'id': 20714}, 'Confiança Consumidor': {'id': 4393}, 'IPCA': {'id': 16122},
                     'Atraso 15-90d Total': {'id': 21006}, 'Atraso 15-90d Agro': {'id': 21069}, 'Inadimplência Crédito Rural': {'id': 21146}}
    lista_dfs_sucesso, config_sucesso = [], {}
    for name, config in SERIES_CONFIG.items():
        try:
            df_temp = sgs.get({name: config['id']}, start='2010-01-01')
            lista_dfs_sucesso.append(df_temp); config_sucesso[name] = config
        except Exception as e: st.warning(f"Não foi possível carregar o indicador '{name}': {e}")
    if not lista_dfs_sucesso: return pd.DataFrame(), {}
    df_full = pd.concat(lista_dfs_sucesso, axis=1); df_full.ffill(inplace=True); df_full.dropna(inplace=True)
    return df_full, config_sucesso

# --- BLOCO 3: LÓGICA DO DASHBOARD DE COMMODITIES (COMPLETO E COM GRÁFICO ÚNICO) ---

# Mapa de commodities movido para o escopo global para ser acessado por múltiplas funções
COMMODITIES_MAP = {
    'Petróleo Brent': 'BZ=F', 'Cacau': 'CC=F', 'Petróleo WTI': 'CL=F', 'Algodão': 'CT=F', 'Ouro': 'GC=F', 'Cobre': 'HG=F', 
    'Óleo de Aquecimento': 'HO=F', 'Café': 'KC=F', 'Trigo (KC HRW)': 'KE=F', 'Madeira': 'LBS=F', 'Gado Bovino': 'LE=F', 
    'Gás Natural': 'NG=F', 'Suco de Laranja': 'OJ=F', 'Paládio': 'PA=F', 'Platina': 'PL=F', 'Gasolina RBOB': 'RB=F',
    'Açúcar': 'SB=F', 'Prata': 'SI=F', 'Milho': 'ZC=F', 'Óleo de Soja': 'ZL=F', 'Aveia': 'ZO=F', 'Arroz': 'ZR=F', 'Soja': 'ZS=F'
}

def obter_precos_atuais(commodities_map):
    """Obtém os preços de mercado mais recentes (sem cache) para uma lista de commodities."""
    precos = {}
    try:
        tickers_data = yf.Tickers(" ".join(commodities_map.values()))
        for nome, ticker_symbol in commodities_map.items():
            info = tickers_data.tickers[ticker_symbol.upper()].info
            if info.get('regularMarketPrice') is not None:
                precos[nome] = info['regularMarketPrice']
            elif info.get('currentPrice') is not None:
                precos[nome] = info['currentPrice']
            else:
                precos[nome] = info.get('previousClose')
    except Exception:
        return {} # Retorna dicionário vazio em caso de falha na API
    return precos

@st.cache_data(ttl=3600*4)
def carregar_dados_commodities():
    dados_commodities_raw = {}
    with st.spinner("Baixando dados históricos de commodities... (cache de 4h)"):
        for nome, ticker in COMMODITIES_MAP.items():
            try:
                dado = yf.download(ticker, period='max', auto_adjust=True, progress=False)
                if not dado.empty: dados_commodities_raw[nome] = dado['Close']
            except Exception: pass
    categorized_commodities = {'Energia': ['Petróleo Brent', 'Petróleo WTI', 'Óleo de Aquecimento', 'Gás Natural', 'Gasolina RBOB'],
                               'Metais Preciosos': ['Ouro', 'Paládio', 'Platina', 'Prata'], 'Metais Industriais': ['Cobre'],
                               'Agricultura': ['Cacau', 'Algodão', 'Café', 'Trigo (KC HRW)', 'Madeira', 'Gado Bovino', 'Suco de Laranja', 
                                              'Açúcar', 'Milho', 'Óleo de Soja', 'Aveia', 'Arroz', 'Soja']}
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
    latest_date, latest_prices = df_full.index.max(), df_full.loc[df_full.index.max()]
    periods = {'1 Dia': 1, '1 Semana': 7, '1 Mês': 30, '3 Meses': 91, '6 Meses': 182, '1 Ano': 365}
    results = []
    for name in df_full.columns:
        res = {'Commodity': name, 'Preço Atual': latest_prices[name]}
        series = df_full[name].dropna()
        for label, days in periods.items():
            past_date = latest_date - timedelta(days=days); past_price = series.asof(past_date)
            res[f'Variação {label}'] = (latest_prices[name] - past_price) / past_price if pd.notna(past_price) and past_price > 0 else np.nan
        results.append(res)
    return pd.DataFrame(results).set_index('Commodity')

def colorir_negativo_positivo(val):
    if pd.isna(val) or val == 0: return ''
    return f"color: {'#4CAF50' if val > 0 else '#F44336'}"

def gerar_dashboard_commodities(dados_preco_por_categoria):
    # (Esta função permanece inalterada)
    all_commodity_names = [name for df in dados_preco_por_categoria.values() for name in df.columns]
    total_subplots = len(all_commodity_names)
    if total_subplots == 0: return go.Figure().update_layout(title_text="Nenhum dado de commodity disponível.")
    num_cols = 4; num_rows = int(np.ceil(total_subplots / num_cols))
    fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=all_commodity_names)
    idx = 0
    for df_cat in dados_preco_por_categoria.values():
        for commodity_name in df_cat.columns:
            row, col = (idx // num_cols) + 1, (idx % num_cols) + 1
            fig.add_trace(go.Scatter(x=df_cat.index, y=df_cat[commodity_name], mode='lines', name=commodity_name), row=row, col=col)
            idx += 1
    end_date = datetime.now(); buttons = []; periods = {'1M': 30, '3M': 91, '6M': 182, 'YTD': 'ytd', '1A': 365, '5A': 365*5, 'Máx': 'max'}
    for label, days in periods.items():
        if days == 'ytd': start_date = datetime(end_date.year, 1, 1)
        elif days == 'max': start_date = min([df.index.min() for df in dados_preco_por_categoria.values() if not df.empty])
        else: start_date = end_date - timedelta(days=days)
        update_args = {}
        for i in range(1, total_subplots + 1):
            update_args[f'xaxis{i if i > 1 else ""}.range'], update_args[f'yaxis{i if i > 1 else ""}.autorange'] = [start_date, end_date], True
        buttons.append(dict(method='relayout', label=label, args=[update_args]))
    active_button_index = list(periods.keys()).index('1A') if '1A' in list(periods.keys()) else 4
    fig.update_layout(title_text="Dashboard de Preços Históricos de Commodities", template="plotly_dark", height=250 * num_rows, showlegend=False,
                      updatemenus=[dict(type="buttons", direction="right", showactive=True, x=1, xanchor="right", y=1.05, yanchor="bottom", buttons=buttons, active=active_button_index)])
    start_date_1y = end_date - timedelta(days=365); idx = 0
    for df_cat in dados_preco_por_categoria.values():
        for commodity_name in df_cat.columns:
            i = idx + 1; fig.layout[f'xaxis{i if i > 1 else ""}.range'] = [start_date_1y, end_date]
            series = df_cat[commodity_name]; filtered_series = series[(series.index >= start_date_1y) & (series.index <= end_date)].dropna()
            if not filtered_series.empty:
                min_y, max_y = filtered_series.min(), filtered_series.max(); padding = (max_y - min_y) * 0.05
                fig.layout[f'yaxis{i if i > 1 else ""}.range'] = [min_y - padding, max_y + padding]
            else: fig.layout[f'yaxis{i if i > 1 else ""}.autorange'] = True
            idx += 1
    return fig

# --- CONSTRUÇÃO DA INTERFACE PRINCIPAL COM ABAS ---
st.title("📊 MOBBT")
st.caption(f"Dados atualizados em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
tab1, tab2, tab3 = st.tabs(["Tesouro Direto", "Indicadores Econômicos (BCB)", "Commodities"])

# --- CONTEÚDO DA ABA 1: TESOURO DIRETO ---
with tab1:
    # (Esta seção permanece inalterada)
    st.header("Análise de Títulos do Tesouro Direto")
    df_tesouro = obter_dados_tesouro()
    if not df_tesouro.empty:
        col1, col2 = st.columns(2)
        with col1:
            tipos_disponiveis = sorted(df_tesouro['Tipo Titulo'].unique())
            tipo_selecionado = st.selectbox("Selecione o Tipo de Título", tipos_disponiveis, key='tipo_tesouro')
        with col2:
            vencimentos_disponiveis = sorted(df_tesouro[df_tesouro['Tipo Titulo'] == tipo_selecionado]['Data Vencimento'].unique())
            vencimento_selecionado = st.selectbox("Selecione a Data de Vencimento", vencimentos_disponiveis, 
                                                  format_func=lambda dt: pd.to_datetime(dt).strftime('%d/%m/%Y'), key='venc_tesouro')
        if vencimento_selecionado:
            st.plotly_chart(gerar_grafico_historico_tesouro(df_tesouro, tipo_selecionado, pd.to_datetime(vencimento_selecionado)), use_container_width=True)
        st.markdown("---")
        st.plotly_chart(gerar_grafico_ettj_curto_prazo(df_tesouro), use_container_width=True)
        st.plotly_chart(gerar_grafico_ettj_longo_prazo(df_tesouro), use_container_width=True)
    else: st.warning("Não foi possível carregar os dados do Tesouro.")

# --- CONTEÚDO DA ABA 2: INDICADORES ECONÔMICOS ---
with tab2:
    # (Esta seção permanece inalterada)
    st.header("Monitor de Indicadores Econômicos do Brasil")
    df_bcb, config_bcb = carregar_dados_bcb()
    if not df_bcb.empty:
        data_inicio = st.date_input("Data de Início", df_bcb.index.min().date(), min_value=df_bcb.index.min().date(), max_value=df_bcb.index.max().date(), key='bcb_start')
        data_fim = st.date_input("Data de Fim", df_bcb.index.max().date(), min_value=data_inicio, max_value=df_bcb.index.max().date(), key='bcb_end')
        df_filtrado_bcb = df_bcb.loc[str(data_inicio):str(data_fim)]
        st.subheader("Gráficos Individuais"); num_cols_bcb = 3; cols_bcb = st.columns(num_cols_bcb)
        for i, nome_serie in enumerate(df_filtrado_bcb.columns):
            fig_bcb = px.line(df_filtrado_bcb, x=df_filtrado_bcb.index, y=nome_serie, title=nome_serie, template='plotly_dark')
            cols_bcb[i % num_cols_bcb].plotly_chart(fig_bcb, use_container_width=True)
    else: st.warning("Não foi possível carregar os dados do BCB.")

# --- CONTEÚDO DA ABA 3: COMMODITIES ---
with tab3:
    st.header("Painel de Preços de Commodities")
    dados_commodities_categorizados = carregar_dados_commodities()

    if dados_commodities_categorizados:
        st.subheader("Variação Percentual de Preços")
        # Calcula a variação com base nos dados históricos (fechamento)
        df_variacao = calcular_variacao_commodities(dados_commodities_categorizados)
        
        # Guarda os preços de fechamento antes de buscar os preços atuais
        precos_fechamento = df_variacao['Preço Atual'].copy()

        # Busca os preços atuais sem cache
        with st.spinner("Buscando cotações atuais..."):
            precos_atuais = obter_precos_atuais(COMMODITIES_MAP)

        # Se a busca por preços atuais funcionou, atualiza a tabela
        if precos_atuais:
            novos_precos_series = pd.Series(precos_atuais)
            # Atualiza a coluna de preço atual
            df_variacao['Preço Atual'].update(novos_precos_series)
            # Recalcula a variação do dia com base no preço atual vs fechamento anterior
            df_variacao['Variação 1 Dia'] = (df_variacao['Preço Atual'] - precos_fechamento) / precos_fechamento

        if not df_variacao.empty:
            cols_variacao = [col for col in df_variacao.columns if 'Variação' in col]
            format_dict = {'Preço Atual': '{:,.2f}'}
            for col in cols_variacao: format_dict[col] = '{:+.2%}'
            
            st.dataframe(df_variacao.style.format(format_dict, na_rep="-")
                                           .applymap(colorir_negativo_positivo, subset=cols_variacao), 
                         use_container_width=True)
            if precos_atuais:
                st.caption("Preços atuais obtidos via Yahoo Finance (podem ter atraso). A variação de 1 dia é calculada em relação ao último fechamento.")
        else:
            st.warning("Não foi possível calcular a variação de preços.")

        st.markdown("---")
        st.plotly_chart(gerar_dashboard_commodities(dados_commodities_categorizados), use_container_width=True)
    else:
        st.warning("Não foi possível carregar os dados de Commodities.")
