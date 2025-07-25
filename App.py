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

# 1.1 - FUNÇÃO DE CARREGAMENTO DE DADOS REATORADA COM CACHE DO STREAMLIT
@st.cache_data(ttl=3600*4) # Cache de 4 horas
def obter_dados_tesouro():
    """
    Carrega dados do Tesouro, usando o cache nativo do Streamlit para performance.
    """
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

# 1.2 - GRÁFICO HISTÓRICO ATUALIZADO PARA ACEITAR TAXA OU PU
def gerar_grafico_historico_tesouro(df, tipo, vencimento, metrica='Taxa Compra Manha'):
    """Gera o gráfico de histórico de taxas OU preço para um título específico."""
    df_filtrado = df[(df['Tipo Titulo'] == tipo) & (df['Data Vencimento'] == vencimento)].sort_values('Data Base')
    
    if metrica == 'Taxa Compra Manha':
        titulo = f'Histórico da Taxa de Compra: {tipo} (Venc. {vencimento.strftime("%d/%m/%Y")})'
        eixo_y = "Taxa de Compra (% a.a.)"
    else:
        titulo = f'Histórico do Preço Unitário (PU): {tipo} (Venc. {vencimento.strftime("%d/%m/%Y")})'
        eixo_y = "Preço Unitário (R$)"
        
    fig = px.line(df_filtrado, x='Data Base', y=metrica, title=titulo, template='plotly_dark')
    fig.update_layout(title_x=0.5, yaxis_title=eixo_y, xaxis_title="Data")
    return fig

# 1.3 - NOVA FUNÇÃO PARA CALCULAR INFLAÇÃO IMPLÍCITA
@st.cache_data
def calcular_inflacao_implicita(df):
    """Calcula a inflação implícita comparando títulos Prefixados e IPCA+."""
    df_recente = df[df['Data Base'] == df['Data Base'].max()].copy()
    
    # Prioriza títulos semestrais por terem mais liquidez, mas usa o principal se não houver
    tipos_ipca = ['Tesouro IPCA+ com Juros Semestrais', 'Tesouro IPCA+']
    df_ipca_raw = df_recente[df_recente['Tipo Titulo'].isin(tipos_ipca)]
    
    df_prefixados = df_recente[df_recente['Tipo Titulo'] == 'Tesouro Prefixado'].set_index('Data Vencimento')
    df_ipca = df_ipca_raw.sort_values('Tipo Titulo', ascending=False).drop_duplicates('Data Vencimento').set_index('Data Vencimento')

    if df_prefixados.empty or df_ipca.empty:
        return pd.DataFrame()

    inflacao_implicita = []
    for venc_prefixado, row_prefixado in df_prefixados.iterrows():
        # Encontra o Tesouro IPCA+ com o vencimento mais próximo
        venc_ipca_proximo = min(df_ipca.index, key=lambda d: abs(d - venc_prefixado))
        
        # Define uma tolerância para o par ser considerado válido (ex: 1 ano e meio)
        if abs((venc_ipca_proximo - venc_prefixado).days) < 550:
            taxa_prefixada = row_prefixado['Taxa Compra Manha']
            row_ipca = df_ipca.loc[venc_ipca_proximo]
            taxa_ipca = row_ipca['Taxa Compra Manha']
            
            # Fórmula de Fisher: ((1 + prefixada) / (1 + real)) - 1
            breakeven = (((1 + taxa_prefixada / 100) / (1 + taxa_ipca / 100)) - 1) * 100
            
            inflacao_implicita.append({
                'Vencimento do Prefixo': venc_prefixado,
                'Inflação Implícita (% a.a.)': breakeven,
                'Par (Venc. IPCA+)': venc_ipca_proximo,
                'Tipo Título IPCA+': row_ipca['Tipo Titulo']
            })
            
    if not inflacao_implicita:
        return pd.DataFrame()

    return pd.DataFrame(inflacao_implicita).sort_values('Vencimento do Prefixo').set_index('Vencimento do Prefixo')

# 1.4 - NOVA FUNÇÃO PARA ANÁLISE DE SPREAD (INCLINAÇÃO DA CURVA)
@st.cache_data
def gerar_grafico_spread_juros(df):
    """Calcula e plota o spread entre o juro prefixado mais longo e o mais curto."""
    df_prefixado = df[df['Tipo Titulo'] == 'Tesouro Prefixado'].copy()
    if df_prefixado.empty:
        return go.Figure()

    vencimentos = sorted(df_prefixado['Data Vencimento'].unique())
    if len(vencimentos) < 2:
        return go.Figure().update_layout(title_text="É necessário no mínimo 2 vencimentos para calcular o spread.")

    venc_curto = vencimentos[0]
    venc_longo = vencimentos[-1]

    df_curto = df_prefixado[df_prefixado['Data Vencimento'] == venc_curto][['Data Base', 'Taxa Compra Manha']].set_index('Data Base')
    df_longo = df_prefixado[df_prefixado['Data Vencimento'] == venc_longo][['Data Base', 'Taxa Compra Manha']].set_index('Data Base')

    df_spread = pd.merge(df_curto, df_longo, on='Data Base', suffixes=('_curto', '_longo')).dropna()
    df_spread['Spread'] = (df_spread['Taxa Compra Manha_longo'] - df_spread['Taxa Compra Manha_curto']) * 100 # Em basis points

    fig = px.area(df_spread, y='Spread', title=f'Spread de Juros: Prefixo {pd.to_datetime(venc_longo).year} vs {pd.to_datetime(venc_curto).year}', template='plotly_dark')
    fig.update_layout(
        title_x=0.5,
        yaxis_title="Diferença (Basis Points)",
        xaxis_title="Data",
        showlegend=False
    )
    return fig


def gerar_grafico_ettj_curto_prazo(df):
    """Gera o gráfico da curva de juros (ETTJ) de CURTO PRAZO para títulos prefixados."""
    df_prefixado = df[df['Tipo Titulo'] == 'Tesouro Prefixado'].copy()
    if df_prefixado.empty:
        return go.Figure().update_layout(title_text="Não há dados para 'Tesouro Prefixado'.")

    datas_disponiveis = sorted(df_prefixado['Data Base'].unique())
    data_recente = datas_disponiveis[-1]

    targets = {
        f'Hoje ({data_recente.strftime("%d/%m/%Y")})': data_recente,
        '1 dia Atrás': data_recente - pd.DateOffset(days=1),
        '2 dias Atrás': data_recente - pd.DateOffset(days=2),
        '3 dias Atrás': data_recente - pd.DateOffset(days=3),
        '4 dias Atrás': data_recente - pd.DateOffset(days=4),
        '5 dias Atrás': data_recente - pd.DateOffset(days=5),
    }

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
                      xaxis_title='Dias Úteis até o Vencimento', yaxis_title='Taxa (% a.a.)',
                      template='plotly_dark', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def gerar_grafico_ettj_longo_prazo(df):
    """Gera o gráfico da curva de juros (ETTJ) de LONGO PRAZO para títulos prefixados."""
    df_prefixado = df[df['Tipo Titulo'] == 'Tesouro Prefixado'].copy()
    if df_prefixado.empty:
        return go.Figure().update_layout(title_text="Não há dados para 'Tesouro Prefixado'.")

    datas_disponiveis = sorted(df_prefixado['Data Base'].unique())
    data_recente = datas_disponiveis[-1]

    targets = {
        f'Hoje ({data_recente.strftime("%d/%m/%Y")})': data_recente,
        '1 Semana Atrás': data_recente - pd.DateOffset(weeks=1),
        '1 Mês Atrás': data_recente - pd.DateOffset(months=1),
        '3 Meses Atrás': data_recente - pd.DateOffset(months=3),
        '6 Meses Atrás': data_recente - pd.DateOffset(months=6),
        '1 Ano Atrás': data_recente - pd.DateOffset(years=1)
    }
    
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
                      xaxis_title='Dias Úteis até o Vencimento', yaxis_title='Taxa (% a.a.)',
                      template='plotly_dark', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


# --- BLOCO 2: LÓGICA DO DASHBOARD DE INDICADORES ECONÔMICOS (EXPANDIDO E ROBUSTO) ---

@st.cache_data(ttl=3600*4) # Cache de 4 horas
def carregar_dados_bcb():
    """Carrega múltiplas séries do SGS do BCB de forma robusta, usando cache."""
    SERIES_CONFIG = {
        'Spread Bancário': {'id': 20783}, 'Inadimplência': {'id': 21082},
        'Crédito/PIB': {'id': 20622}, 'Juros Médio': {'id': 20714},
        'Confiança Consumidor': {'id': 4393}, 'IPCA': {'id': 16122},
        'Atraso 15-90d Total': {'id': 21006},
        'Atraso 15-90d Agro': {'id': 21069},
        'Inadimplência Crédito Rural': {'id': 21146},
    }
    
    lista_dfs_sucesso = []
    config_sucesso = {}
    
    for name, config in SERIES_CONFIG.items():
        try:
            df_temp = sgs.get({name: config['id']}, start='2010-01-01')
            lista_dfs_sucesso.append(df_temp)
            config_sucesso[name] = config
        except Exception as e:
            st.warning(f"Não foi possível carregar o indicador '{name}': {e}")

    if not lista_dfs_sucesso:
        return pd.DataFrame(), {}
        
    df_full = pd.concat(lista_dfs_sucesso, axis=1)
    df_full.ffill(inplace=True)
    df_full.dropna(inplace=True)
    return df_full, config_sucesso


# --- BLOCO 3: LÓGICA DO DASHBOARD DE COMMODITIES (COMPLETO E COM GRÁFICO ÚNICO) ---

@st.cache_data(ttl=3600*4) # Cache de 4 horas
def carregar_dados_commodities():
    """Baixa e categoriza uma lista completa de commodities do Yahoo Finance."""
    commodities_map = {
        'Petróleo Brent': 'BZ=F', 'Cacau': 'CC=F', 'Petróleo WTI': 'CL=F', 'Algodão': 'CT=F',
        'Ouro': 'GC=F', 'Cobre': 'HG=F', 'Óleo de Aquecimento': 'HO=F', 'Café': 'KC=F',
        'Trigo (KC HRW)': 'KE=F', 'Madeira': 'LBS=F', 'Gado Bovino': 'LE=F', 'Gás Natural': 'NG=F',
        'Suco de Laranja': 'OJ=F', 'Paládio': 'PA=F', 'Platina': 'PL=F', 'Gasolina RBOB': 'RB=F',
        'Açúcar': 'SB=F', 'Prata': 'SI=F', 'Milho': 'ZC=F', 'Óleo de Soja': 'ZL=F',
        'Aveia': 'ZO=F', 'Arroz': 'ZR=F', 'Soja': 'ZS=F'
    }
    
    dados_commodities_raw = {}
    with st.spinner("Baixando dados históricos de commodities... (cache de 4h)"):
        for nome, ticker in commodities_map.items():
            try:
                dado = yf.download(ticker, period='max', auto_adjust=True, progress=False)
                if not dado.empty:
                    dados_commodities_raw[nome] = dado['Close']
            except Exception:
                pass 

    categorized_commodities = {
        'Energia': ['Petróleo Brent', 'Petróleo WTI', 'Óleo de Aquecimento', 'Gás Natural', 'Gasolina RBOB'],
        'Metais Preciosos': ['Ouro', 'Paládio', 'Platina', 'Prata'],
        'Metais Industriais': ['Cobre'],
        'Agricultura': [
            'Cacau', 'Algodão', 'Café', 'Trigo (KC HRW)', 'Madeira', 'Gado Bovino', 'Suco de Laranja',
            'Açúcar', 'Milho', 'Óleo de Soja', 'Aveia', 'Arroz', 'Soja'
        ]
    }
    
    dados_por_categoria = {}
    for categoria, nomes in categorized_commodities.items():
        series_da_categoria = {
            nome: dados_commodities_raw[nome] 
            for nome in nomes 
            if nome in dados_commodities_raw
        }
        
        if series_da_categoria:
            df_cat = pd.concat(series_da_categoria, axis=1)
            df_cat.columns = series_da_categoria.keys()
            dados_por_categoria[categoria] = df_cat

    return dados_por_categoria

def calcular_variacao_commodities(dados_por_categoria):
    """Calcula a variação de preços em diferentes períodos para todas as commodities."""
    all_series = []
    for df_cat in dados_por_categoria.values():
        for col in df_cat.columns:
            all_series.append(df_cat[col].dropna())

    if not all_series:
        return pd.DataFrame()

    df_full = pd.concat(all_series, axis=1)
    df_full.sort_index(inplace=True)

    if df_full.empty:
        return pd.DataFrame()

    latest_date = df_full.index.max()
    latest_prices = df_full.loc[latest_date]

    periods = {
        '1 Dia': 1, '1 Semana': 7, '1 Mês': 30,
        '3 Meses': 91, '6 Meses': 182, '1 Ano': 365
    }

    results = []
    for commodity_name in df_full.columns:
        res = {'Commodity': commodity_name, 'Preço Atual': latest_prices[commodity_name]}
        commodity_series = df_full[commodity_name].dropna()

        for period_label, days_ago in periods.items():
            past_date = latest_date - timedelta(days=days_ago)
            past_price = commodity_series.asof(past_date)

            if pd.notna(past_price) and past_price > 0:
                variation = ((latest_prices[commodity_name] - past_price) / past_price)
            else:
                variation = np.nan
            res[f'Variação {period_label}'] = variation
        results.append(res)

    df_results = pd.DataFrame(results).set_index('Commodity')
    return df_results

def colorir_negativo_positivo(val):
    """Aplica cor verde para valores positivos e vermelho para negativos."""
    if pd.isna(val) or val == 0:
        return ''
    color = '#4CAF50' if val > 0 else '#F44336' # Verde e Vermelho
    return f'color: {color}'

def gerar_dashboard_commodities(dados_preco_por_categoria):
    """Cria um único dashboard com subplots, botões de período e escala do eixo Y dinâmica."""
    if not dados_preco_por_categoria:
        return go.Figure().update_layout(title_text="Nenhuma commodity pôde ser carregada.")

    all_commodity_names = [name for df in dados_preco_por_categoria.values() for name in df.columns]
    total_subplots = len(all_commodity_names)
    
    if total_subplots == 0:
        return go.Figure().update_layout(title_text="Nenhum dado de commodity disponível.")
        
    num_cols = 4
    num_rows = int(np.ceil(total_subplots / num_cols))
    fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=all_commodity_names)
    
    idx = 0
    for df_cat in dados_preco_por_categoria.values():
        for commodity_name in df_cat.columns:
            row = (idx // num_cols) + 1
            col = (idx % num_cols) + 1
            fig.add_trace(go.Scatter(x=df_cat.index, y=df_cat[commodity_name], mode='lines', name=commodity_name), row=row, col=col)
            idx += 1
            
    end_date = datetime.now()
    buttons = []
    periods = {'1M': 30, '3M': 91, '6M': 182, 'YTD': 'ytd', '1A': 365, '5A': 365*5, 'Máx': 'max'}
    
    for label, days in periods.items():
        if days == 'ytd':
            start_date = datetime(end_date.year, 1, 1)
        elif days == 'max':
            start_date = min([df.index.min() for df in dados_preco_por_categoria.values() if not df.empty])
        else:
            start_date = end_date - timedelta(days=days)
        
        update_args = {}
        for i in range(1, total_subplots + 1):
            xaxis_name = f'xaxis{i}' if i > 1 else 'xaxis'
            yaxis_name = f'yaxis{i}' if i > 1 else 'yaxis'
            update_args[f'{xaxis_name}.range'] = [start_date, end_date]
            update_args[f'{yaxis_name}.autorange'] = True 

        buttons.append(dict(method='relayout', label=label, args=[update_args]))

    period_labels = list(periods.keys())
    active_button_index = period_labels.index('1A') if '1A' in period_labels else 4

    fig.update_layout(
        title_text="Dashboard de Preços Históricos de Commodities",
        template="plotly_dark",
        height=250 * num_rows,
        showlegend=False,
        updatemenus=[
            dict(type="buttons", direction="right", showactive=True,
                 x=1, xanchor="right", y=1.05, yanchor="bottom", buttons=buttons,
                 active=active_button_index)
        ]
    )
    
    start_date_1y = end_date - timedelta(days=365)
    
    idx = 0
    for df_cat in dados_preco_por_categoria.values():
        for commodity_name in df_cat.columns:
            i = idx + 1
            xaxis_name = f'xaxis{i}' if i > 1 else 'xaxis'
            yaxis_name = f'yaxis{i}' if i > 1 else 'yaxis'

            fig.layout[xaxis_name].range = [start_date_1y, end_date]

            series = df_cat[commodity_name]
            filtered_series = series[(series.index >= start_date_1y) & (series.index <= end_date)].dropna()

            if not filtered_series.empty:
                min_y = filtered_series.min()
                max_y = filtered_series.max()
                padding = (max_y - min_y) * 0.05
                fig.layout[yaxis_name].range = [min_y - padding, max_y + padding]
            else:
                fig.layout[yaxis_name].autorange = True
            
            idx += 1

    return fig

# --- CONSTRUÇÃO DA INTERFACE PRINCIPAL COM ABAS ---

st.title("📊 MOBBT")
st.caption(f"Dados atualizados em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

tab1, tab2, tab3 = st.tabs(["Tesouro Direto", "Indicadores Econômicos (BCB)", "Commodities"])

# --- CONTEÚDO DA ABA 1: TESOURO DIRETO (ATUALIZADO) ---
with tab1:
    st.header("Análise de Títulos do Tesouro Direto")
    df_tesouro = obter_dados_tesouro()

    if not df_tesouro.empty:
        st.subheader("Análise Histórica de Título Individual")
        col1, col2 = st.columns(2)
        with col1:
            tipos_disponiveis = sorted(df_tesouro['Tipo Titulo'].unique())
            tipo_selecionado = st.selectbox("Selecione o Tipo de Título", tipos_disponiveis, key='tipo_tesouro')
        with col2:
            vencimentos_disponiveis = sorted(df_tesouro[df_tesouro['Tipo Titulo'] == tipo_selecionado]['Data Vencimento'].unique())
            vencimento_selecionado = st.selectbox(
                "Selecione a Data de Vencimento", vencimentos_disponiveis,
                format_func=lambda dt: pd.to_datetime(dt).strftime('%d/%m/%Y'), key='venc_tesouro'
            )
        
        # Seletor para Taxa ou PU
        metrica_escolhida = st.radio(
            "Analisar por:", ('Taxa de Compra', 'Preço Unitário (PU)'), 
            horizontal=True, key='metrica_tesouro',
            help="**Taxa de Compra:** Rentabilidade anual do título. **Preço Unitário:** Valor do título, útil para ver o efeito da marcação a mercado."
        )
        coluna_metrica = 'Taxa Compra Manha' if metrica_escolhida == 'Taxa de Compra' else 'PU Compra Manha'

        if vencimento_selecionado:
            fig_historico = gerar_grafico_historico_tesouro(
                df_tesouro, tipo_selecionado, pd.to_datetime(vencimento_selecionado), metrica=coluna_metrica
            )
            st.plotly_chart(fig_historico, use_container_width=True)

        st.markdown("---")
        
        st.subheader("Análises da Curva de Juros")
        # As duas colunas agora abrigam as análises de spread e inflação implícita
        col_analise1, col_analise2 = st.columns(2)

        with col_analise1:
            st.info("A **Inflação Implícita** mostra a expectativa do mercado para a inflação futura, extraída da diferença entre as taxas de títulos prefixados e atrelados ao IPCA.")
            df_breakeven = calcular_inflacao_implicita(df_tesouro)
            if not df_breakeven.empty:
                st.dataframe(
                    df_breakeven[['Inflação Implícita (% a.a.)']].style.format({'Inflação Implícita (% a.a.)': '{:.2f}%'}),
                    use_container_width=True
                )
                fig_breakeven = px.bar(
                    df_breakeven, y='Inflação Implícita (% a.a.)', text_auto='.2f',
                    title='Inflação Implícita por Vencimento'
                )
                fig_breakeven.update_traces(textposition='outside')
                st.plotly_chart(fig_breakeven, use_container_width=True)
            else:
                st.warning("Não há pares de títulos disponíveis hoje para calcular a inflação implícita.")

        with col_analise2:
            st.info("O **Spread de Juros** mostra a diferença entre as taxas de um título longo e um curto. Uma diferença positiva (inclinação) sugere otimismo econômico.")
            fig_spread = gerar_grafico_spread_juros(df_tesouro)
            st.plotly_chart(fig_spread, use_container_width=True)

        st.markdown("---")
        
        # Gráficos da ETTJ mantidos no final
        st.subheader("Estrutura a Termo da Taxa de Juros (ETTJ) - Títulos Prefixados")
        fig_ettj_curto = gerar_grafico_ettj_curto_prazo(df_tesouro)
        st.plotly_chart(fig_ettj_curto, use_container_width=True)

        fig_ettj_longo = gerar_grafico_ettj_longo_prazo(df_tesouro)
        st.plotly_chart(fig_ettj_longo, use_container_width=True)
        
    else:
        st.warning("Não foi possível carregar os dados do Tesouro.")

# --- CONTEÚDO DA ABA 2: INDICADORES ECONÔMICOS ---
with tab2:
    st.header("Monitor de Indicadores Econômicos do Brasil")
    df_bcb, config_bcb = carregar_dados_bcb()

    if not df_bcb.empty:
        data_inicio = st.date_input("Data de Início", df_bcb.index.min().date(), min_value=df_bcb.index.min().date(), max_value=df_bcb.index.max().date(), key='bcb_start')
        data_fim = st.date_input("Data de Fim", df_bcb.index.max().date(), min_value=data_inicio, max_value=df_bcb.index.max().date(), key='bcb_end')

        df_filtrado_bcb = df_bcb.loc[str(data_inicio):str(data_fim)]
        
        st.subheader("Gráficos Individuais")
        num_cols_bcb = 3
        cols_bcb = st.columns(num_cols_bcb)
        for i, nome_serie in enumerate(df_filtrado_bcb.columns):
            fig_bcb = px.line(df_filtrado_bcb, x=df_filtrado_bcb.index, y=nome_serie, title=nome_serie, template='plotly_dark')
            cols_bcb[i % num_cols_bcb].plotly_chart(fig_bcb, use_container_width=True)
    else:
        st.warning("Não foi possível carregar os dados do BCB.")

# --- CONTEÚDO DA ABA 3: COMMODITIES ---
with tab3:
    st.header("Painel de Preços de Commodities")
    dados_commodities_categorizados = carregar_dados_commodities()

    if dados_commodities_categorizados:
        # Tabela de variação exibida primeiro
        st.subheader("Variação Percentual de Preços")
        df_variacao = calcular_variacao_commodities(dados_commodities_categorizados)
        
        if not df_variacao.empty:
            # Define as colunas de variação para aplicar a formatação e as cores
            cols_variacao = [col for col in df_variacao.columns if 'Variação' in col]
            
            # Cria o dicionário de formatação dinamicamente
            format_dict = {'Preço Atual': '{:,.2f}'}
            for col in cols_variacao:
                format_dict[col] = '{:+.2%}'

            # Aplica a formatação de números e a de cores
            st.dataframe(df_variacao.style.format(format_dict, na_rep="-")
                                           .applymap(colorir_negativo_positivo, subset=cols_variacao), 
                         use_container_width=True)
        else:
            st.warning("Não foi possível calcular a variação de preços.")

        st.markdown("---")
        
        # Gráficos de preços históricos exibidos depois da tabela
        fig_commodities = gerar_dashboard_commodities(dados_commodities_categorizados)
        st.plotly_chart(fig_commodities, use_container_width=True)
    else:
        st.warning("Não foi possível carregar os dados de Commodities.")
