
import streamlit as st
import pandas as pd
from datetime import datetime
from src.data_loaders.cvm import baixar_e_extrair_zip_cvm
from src.models.insiders import (
    analisar_dados_insiders, criar_lookup_ticker_cnpj, 
    analisar_historico_insider_por_ticker, obter_detalhes_insider_por_ticker,
    analisar_historico_insider_por_nome, obter_detalhes_insider_por_nome
)
from src.components.charts_insiders import gerar_grafico_historico_insider

def render():
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
    with st.spinner(f"Baixando e pré-processando dados da CVM ({ANO_ATUAL})..."):
        df_mov_bruto = baixar_e_extrair_zip_cvm(URL_MOVIMENTACOES, CSV_MOVIMENTACOES, show_error=False)
        df_cad_bruto = baixar_e_extrair_zip_cvm(URL_CADASTRO, CSV_CADASTRO, show_error=False)

    if df_mov_bruto is None or df_cad_bruto is None:
        ANO_ANTERIOR = ANO_ATUAL - 1
        st.warning(f"Dados de {ANO_ATUAL} ainda não disponíveis na CVM. Utilizando dados de {ANO_ANTERIOR}...")
        
        URL_MOVIMENTACOES = f"https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/VLMO/DADOS/vlmo_cia_aberta_{ANO_ANTERIOR}.zip"
        URL_CADASTRO = f"https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FCA/DADOS/fca_cia_aberta_{ANO_ANTERIOR}.zip"
        CSV_MOVIMENTACOES = f"vlmo_cia_aberta_con_{ANO_ANTERIOR}.csv"
        CSV_CADASTRO = f"fca_cia_aberta_valor_mobiliario_{ANO_ANTERIOR}.csv"
        
        with st.spinner(f"Baixando dados do ano anterior ({ANO_ANTERIOR})..."):
            df_mov_bruto = baixar_e_extrair_zip_cvm(URL_MOVIMENTACOES, CSV_MOVIMENTACOES, show_error=True)
            df_cad_bruto = baixar_e_extrair_zip_cvm(URL_CADASTRO, CSV_CADASTRO, show_error=True)

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
                    '% do Market Cap': '{:.4f}%',
                    'Preço Médio Compras (R$)': 'R$ {:,.2f}'
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

        st.markdown("---")
        st.subheader("Analisar Histórico Detalhado por Ticker")
        st.info("Digite o código de negociação (ex: PETR4, ALOS3).")

        lookup_ticker_cnpj = criar_lookup_ticker_cnpj(df_cad_bruto)

        col_search_1, col_search_2 = st.columns([0.8, 0.2])
        with col_search_1:
            ticker_input = st.text_input("Digite o Ticker:", key="insider_ticker_input", placeholder="Ex: PETR4").upper().strip()
        with col_search_2:
            st.write(""); st.write("")
            btn_buscar = st.button("Buscar", use_container_width=True)

        if btn_buscar and ticker_input:
            cnpj_alvo = lookup_ticker_cnpj.get(ticker_input)
            
            if not cnpj_alvo:
                st.error(f"Ticker '{ticker_input}' não encontrado. Adicione ao dicionário manual se necessário.")
            else:
                df_historico_grafico = analisar_historico_insider_por_ticker(df_mov_bruto, cnpj_alvo)
                fig_hist = gerar_grafico_historico_insider(df_historico_grafico, ticker_input)
                st.plotly_chart(fig_hist, use_container_width=True)
                
                detalhes = obter_detalhes_insider_por_ticker(df_mov_bruto, cnpj_alvo)
                if not detalhes.empty:
                    st.dataframe(detalhes.style.format({'Preço (R$)': 'R$ {:.2f}', 'Volume Total (R$)': 'R$ {:,.2f}'}), use_container_width=True)
                else:
                    st.info("Sem transações detalhadas disponíveis.")

        st.markdown("---")
        st.subheader("Analisar Histórico Detalhado por Empresa (Busca por Nome)")
        st.info("Digite parte do nome da empresa para buscar (ex: 'Petrobras', 'Magazine'). Útil se o ticker não for encontrado.")

        col_search_nm_1, col_search_nm_2 = st.columns([0.8, 0.2])
        with col_search_nm_1:
             nome_input = st.text_input("Nome da Empresa:", key="insider_nome_input", placeholder="Ex: Petrobras").strip()
        with col_search_nm_2:
             st.write(""); st.write("")
             btn_buscar_nome = st.button("Buscar por Nome", use_container_width=True)

        if btn_buscar_nome and nome_input:
             df_historico_grafico = analisar_historico_insider_por_nome(df_mov_bruto, nome_input)
             if not df_historico_grafico.empty:
                 fig_hist = gerar_grafico_historico_insider(df_historico_grafico, nome_input)
                 st.plotly_chart(fig_hist, use_container_width=True)
                 
                 detalhes = obter_detalhes_insider_por_nome(df_mov_bruto, nome_input)
                 st.dataframe(detalhes.style.format({'Preço (R$)': 'R$ {:.2f}', 'Volume Total (R$)': 'R$ {:,.2f}'}), use_container_width=True)
             else:
                 st.warning(f"Nenhuma empresa encontrada com o nome contendo '{nome_input}'.")
    else:
        st.error("Falha fatal ao carregar dados da CVM.")
