
import pandas as pd
import numpy as np
import streamlit as st
import re
from src.data_loaders.stocks import buscar_market_caps_otimizado

@st.cache_data
def analisar_dados_insiders(_df_mov, _df_cad, meses_selecionados, force_refresh=False):
    """
    Análise de insiders com correção manual FORÇADA de Tickers.
    """
    if not meses_selecionados:
        return pd.DataFrame()

    # --- 1. Filtro de Movimentações ---
    df_periodo = _df_mov[_df_mov['Ano_Mes'].isin(meses_selecionados)].copy()
    if df_periodo.empty:
        return pd.DataFrame()

    tipos_compra = ['Compra à vista', 'Recompra', 'Recompra de ações']
    tipos_venda = ['Venda à vista']
    
    df_periodo = df_periodo[df_periodo['Tipo_Movimentacao'].isin(tipos_compra + tipos_venda)].copy()

    df_periodo['Volume_Net'] = np.where(
        df_periodo['Tipo_Movimentacao'].isin(tipos_compra), 
        df_periodo['Volume'], 
        -df_periodo['Volume']
    )

    # Agrupa por CNPJ
    df_net_total = df_periodo.groupby(['CNPJ_Companhia', 'Nome_Companhia'])['Volume_Net'].sum().reset_index()

    # --- 2. Normalização de CNPJ ---
    def limpar_cnpj(series):
        return series.astype(str).str.replace(r'[./-]', '', regex=True).str.strip()

    df_net_total['CNPJ_Limpo'] = limpar_cnpj(df_net_total['CNPJ_Companhia'])
    
    # Prepara cadastro (FCA)
    df_cad_valido = _df_cad.copy()
    if 'Valor_Mobiliario' in df_cad_valido.columns:
        filtro = df_cad_valido['Valor_Mobiliario'].astype(str).str.contains('Aç|Ac', case=False, na=False)
        if filtro.any():
            df_cad_valido = df_cad_valido[filtro]
            
    df_cad_valido['CNPJ_Limpo'] = limpar_cnpj(df_cad_valido['CNPJ_Companhia'])
    
    # Lookup: CNPJ -> Ticker
    df_tickers = df_cad_valido[['CNPJ_Limpo', 'Codigo_Negociacao']].dropna().drop_duplicates(subset=['CNPJ_Limpo'])

    # --- 3.5 Cálculo do Preço Médio de Compra ---
    df_compras = df_periodo[df_periodo['Tipo_Movimentacao'].isin(tipos_compra)].copy()
    
    if not df_compras.empty:
        df_pm_compras = df_compras.groupby(['CNPJ_Companhia'])[['Volume', 'Quantidade']].sum().reset_index()
        df_pm_compras['Preco_Medio_Compra'] = np.where(
            df_pm_compras['Quantidade'] > 0,
            df_pm_compras['Volume'] / df_pm_compras['Quantidade'],
            0
        )
        df_pm_compras['CNPJ_Limpo'] = limpar_cnpj(df_pm_compras['CNPJ_Companhia'])
        df_pm_compras = df_pm_compras[['CNPJ_Limpo', 'Preco_Medio_Compra']]
    else:
        df_pm_compras = pd.DataFrame(columns=['CNPJ_Limpo', 'Preco_Medio_Compra'])

    # --- 4. Merge e CORREÇÃO FORÇADA ---
    df_merged = pd.merge(df_net_total, df_tickers, on='CNPJ_Limpo', how='left')
    df_merged = pd.merge(df_merged, df_pm_compras, on='CNPJ_Limpo', how='left')

    correcoes_manuais = {
        '05878397000132': 'ALOS3',  # Allos
        '59717553000102': 'MLAS3',  # Grupo Multi
        '08312229000173': 'EZTC3',  # EZTec
        '00001180000126': 'ELET3',  # Eletrobras
        '61088894000108': 'CAMB3',  # Cambuci
        '60651809000105': 'SUZB3',  # Suzano
        '61156113000175': 'MYPK3',  # Iochpe
        '28127603000178': 'BEES3',  # Banestes
        '42771949000135': 'QUAL3',  # Aliança/Qualicorp
        '50746577000115': 'CSAN3',  # COSAN
    }
    
    for cnpj, ticker in correcoes_manuais.items():
        df_merged.loc[df_merged['CNPJ_Limpo'] == cnpj, 'Codigo_Negociacao'] = ticker

    df_merged['Codigo_Negociacao'] = df_merged['Codigo_Negociacao'].fillna("SEM_TICKER")
    df_merged['Codigo_Negociacao'] = df_merged['Codigo_Negociacao'].replace('', 'SEM_TICKER')

    # --- 5. Market Cap ---
    df_lookup_mcap = df_merged[df_merged['Codigo_Negociacao'] != "SEM_TICKER"][['Codigo_Negociacao']].drop_duplicates()
    df_market_cap_lookup = buscar_market_caps_otimizado(df_lookup_mcap, force_refresh=force_refresh)

    df_final = pd.merge(df_merged, df_market_cap_lookup[['Codigo_Negociacao', 'MarketCap']], on='Codigo_Negociacao', how='left')
    
    market_cap_para_calculo = df_final['MarketCap'].fillna(0)
    df_final['Volume_vs_MarketCap_Pct'] = np.where(
        market_cap_para_calculo > 0,
        (df_final['Volume_Net'] / market_cap_para_calculo) * 100,
        0
    )

    df_tabela = df_final[[
        'Codigo_Negociacao', 'Nome_Companhia', 'Volume_Net', 'MarketCap', 'Volume_vs_MarketCap_Pct', 'Preco_Medio_Compra', 'CNPJ_Companhia'
    ]].rename(columns={
        'Codigo_Negociacao': 'Ticker', 'Nome_Companhia': 'Empresa',
        'Volume_Net': 'Volume Líquido (R$)', 'MarketCap': 'Valor de Mercado (R$)',
        'Volume_vs_MarketCap_Pct': '% do Market Cap', 'Preco_Medio_Compra': 'Preço Médio Compras (R$)'
    })

    return df_tabela.sort_values(by='Volume Líquido (R$)', ascending=False).reset_index(drop=True)

@st.cache_data
def criar_lookup_ticker_cnpj(_df_cad):
    df_tickers = _df_cad[['CNPJ_Companhia', 'Codigo_Negociacao']].dropna()
    df_tickers = df_tickers.drop_duplicates(subset=['Codigo_Negociacao'])
    
    lookup = pd.Series(df_tickers['CNPJ_Companhia'].values, index=df_tickers['Codigo_Negociacao']).to_dict()
    
    correcoes_busca = {
        'ALOS3': '05.878.397/0001-32',
        'MLAS3': '59.717.553/0001-02',
    }
    lookup.update(correcoes_busca)
    return lookup

@st.cache_data
def analisar_historico_insider_por_nome(_df_mov, nome_alvo):
    if not nome_alvo or _df_mov.empty:
        return pd.DataFrame()

    def normalizar(s): return str(s).upper().replace('.', '').replace(',', '').strip()
    nome_alvo_norm = normalizar(nome_alvo)
    
    df_temp = _df_mov.copy()
    df_temp['Nome_Norm'] = df_temp['Nome_Companhia'].apply(normalizar)
    
    df_empresa = df_temp[df_temp['Nome_Norm'].str.contains(re.escape(nome_alvo_norm), na=False)].copy()

    if df_empresa.empty:
        return pd.DataFrame()

    tipos_compra = ['Compra à vista', 'Recompra', 'Recompra de ações']
    tipos_venda = ['Venda à vista']
    df_empresa = df_empresa[df_empresa['Tipo_Movimentacao'].isin(tipos_compra + tipos_venda)]

    df_empresa['Volume_Net'] = np.where(
        df_empresa['Tipo_Movimentacao'].isin(tipos_compra),
        df_empresa['Volume'],
        -df_empresa['Volume']
    )

    df_historico = df_empresa.groupby('Ano_Mes')['Volume_Net'].sum().reset_index()
    df_historico = df_historico.sort_values(by='Ano_Mes')
    df_historico['Data'] = pd.to_datetime(df_historico['Ano_Mes'] + '-01')

    return df_historico[['Data', 'Volume_Net']]

@st.cache_data
def obter_detalhes_insider_por_nome(_df_mov, nome_alvo):
    if not nome_alvo or _df_mov.empty: return pd.DataFrame()

    def normalizar(s): return str(s).upper().replace('.', '').replace(',', '').strip()
    nome_alvo_norm = normalizar(nome_alvo)
    
    df_temp = _df_mov.copy()
    df_temp['Nome_Norm'] = df_temp['Nome_Companhia'].apply(normalizar)
    df_detalhes = df_temp[df_temp['Nome_Norm'].str.contains(re.escape(nome_alvo_norm), na=False)].copy()

    operacoes_validas = ['Compra à vista', 'Venda à vista', 'Recompra', 'Recompra de ações']
    df_detalhes = df_detalhes[df_detalhes['Tipo_Movimentacao'].isin(operacoes_validas)]

    if df_detalhes.empty: return pd.DataFrame()

    colunas_desejadas = {
        'Data_Movimentacao': 'Data', 'Tipo_Cargo': 'Cargo / Grupo', 
        'Tipo_Movimentacao': 'Operação', 'Quantidade': 'Qtd.',
        'Preco_Unitario': 'Preço (R$)', 'Volume': 'Volume Total (R$)'
    }
    
    cols_existentes = [c for c in colunas_desejadas.keys() if c in df_detalhes.columns]
    df_exibicao = df_detalhes[cols_existentes].rename(columns=colunas_desejadas)

    if 'Data' in df_exibicao.columns:
        df_exibicao = df_exibicao.sort_values(by='Data', ascending=False)

    return df_exibicao

@st.cache_data
def analisar_historico_insider_por_ticker(_df_mov, cnpj_alvo):
    if not cnpj_alvo or _df_mov.empty: return pd.DataFrame()
    
    df_empresa = _df_mov[_df_mov['CNPJ_Companhia'] == cnpj_alvo].copy()
    if df_empresa.empty: return pd.DataFrame()

    tipos_compra = ['Compra à vista', 'Recompra', 'Recompra de ações']
    tipos_venda = ['Venda à vista']
    df_empresa = df_empresa[df_empresa['Tipo_Movimentacao'].isin(tipos_compra + tipos_venda)]

    df_empresa['Volume_Net'] = np.where(
        df_empresa['Tipo_Movimentacao'].isin(tipos_compra),
        df_empresa['Volume'], -df_empresa['Volume']
    )

    df_historico = df_empresa.groupby('Ano_Mes')['Volume_Net'].sum().reset_index().sort_values('Ano_Mes')
    df_historico['Data'] = pd.to_datetime(df_historico['Ano_Mes'] + '-01')
    return df_historico[['Data', 'Volume_Net']]

@st.cache_data
def obter_detalhes_insider_por_ticker(_df_mov, cnpj_alvo):
    if not cnpj_alvo or _df_mov.empty: return pd.DataFrame()
    
    df_detalhes = _df_mov[_df_mov['CNPJ_Companhia'] == cnpj_alvo].copy()
    operacoes_validas = ['Compra à vista', 'Venda à vista', 'Recompra', 'Recompra de ações']
    df_detalhes = df_detalhes[df_detalhes['Tipo_Movimentacao'].isin(operacoes_validas)]
    
    colunas = {'Data_Movimentacao': 'Data', 'Tipo_Cargo': 'Cargo / Grupo', 'Tipo_Movimentacao': 'Operação', 
               'Quantidade': 'Qtd.', 'Preco_Unitario': 'Preço (R$)', 'Volume': 'Volume Total (R$)'}
    
    existentes = [c for c in colunas.keys() if c in df_detalhes.columns]
    return df_detalhes[existentes].rename(columns=colunas).sort_values('Data', ascending=False)
