"""
Página de Exportação de DFPs - Demonstrações Financeiras Padronizadas
Permite buscar e exportar dados financeiros de empresas da CVM
"""

import streamlit as st
import pandas as pd
import io
import sys
from pathlib import Path
from datetime import datetime

# Adiciona o diretório raiz ao path para importar DFP
ROOT_DIR = Path(__file__).parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Importa funções do módulo DFP
from DFP import (
    carregar_cadastro_com_cache,
    baixar_e_processar_em_paralelo,
    formatar_demonstrativo,
    combinar_dados,
    URL_BASE_CVM
)


def render():
    """Renderiza a página de exportação de DFPs."""
    
    st.header("📊 Exportador de DFPs")
    st.markdown("Exporte demonstrativos financeiros (DRE, Balanço, DFC) de empresas listadas na CVM.")
    st.markdown("---")
    
    # --- Carregamento do Cadastro de Empresas ---
    @st.cache_data(ttl=3600*24, show_spinner=False)
    def carregar_empresas():
        """Carrega e cacheia lista de empresas ativas da CVM."""
        return carregar_cadastro_com_cache()
    
    with st.spinner("Carregando lista de empresas..."):
        try:
            df_empresas = carregar_empresas()
        except Exception as e:
            st.error(f"Erro ao carregar cadastro de empresas: {e}")
            return
    
    if df_empresas.empty:
        st.warning("Nenhuma empresa ativa encontrada no cadastro CVM.")
        return
    
    # --- Interface de Seleção ---
    col1, col2 = st.columns([3, 1])
    
    with col1:
        busca = st.text_input(
            "🔍 Buscar empresa",
            placeholder="Digite o nome da empresa (mín. 3 caracteres)...",
            help="Busque pelo nome da empresa. Ex: PETROBRAS, VALE, ITAU"
        )
    
    empresa_selecionada = None
    codigo_cvm = None
    
    if busca and len(busca.strip()) >= 3:
        query = busca.strip().upper()
        resultados = df_empresas[
            df_empresas['DENOM_SOCIAL'].str.contains(query, case=False, na=False)
        ].head(20)
        
        if not resultados.empty:
            opcoes = resultados['DENOM_SOCIAL'].tolist()
            
            empresa_selecionada = st.selectbox(
                "Selecione a empresa:",
                options=opcoes,
                help="Escolha a empresa para exportar os dados financeiros"
            )
            
            if empresa_selecionada:
                codigo_cvm = int(resultados[
                    resultados['DENOM_SOCIAL'] == empresa_selecionada
                ]['CD_CVM'].values[0])
                
                st.info(f"**Empresa:** {empresa_selecionada} | **Código CVM:** {codigo_cvm}")
        else:
            st.warning(f"Nenhuma empresa encontrada para '{query}'.")
    elif busca and len(busca.strip()) < 3:
        st.caption("Digite pelo menos 3 caracteres para buscar.")
    
    st.markdown("---")
    
    # --- Configurações de Exportação ---
    st.subheader("⚙️ Configurações")
    
    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    
    with col_cfg1:
        anos_historico = st.slider(
            "Anos de histórico (DFP)",
            min_value=5,
            max_value=20,
            value=10,
            help="Quantidade de anos de dados anuais (DFP) a buscar"
        )
    
    with col_cfg2:
        anos_trimestral = st.slider(
            "Anos de ITRs",
            min_value=1,
            max_value=5,
            value=3,
            help="Quantidade de anos de dados trimestrais (ITR) a buscar"
        )
    
    with col_cfg3:
        tipo_dados = st.radio(
            "Tipo de dados",
            options=["Consolidado", "Individual", "Ambos"],
            index=0,
            help="Consolidado: inclui subsidiárias. Individual: apenas a empresa matriz."
        )
    
    st.markdown("---")
    
    # --- Botão de Exportação ---
    if empresa_selecionada and codigo_cvm:
        if st.button("📥 Gerar e Baixar Excel", type="primary", use_container_width=True):
            with st.status("Processando dados financeiros...", expanded=True) as status:
                try:
                    ano_atual = datetime.now().year
                    
                    # Buscar dados DFP (anuais)
                    st.write("📅 Baixando dados anuais (DFP)...")
                    anos_dfp = list(range(ano_atual - anos_historico, ano_atual + 1))
                    dados_dfp = baixar_e_processar_em_paralelo(anos_dfp, codigo_cvm, 'DFP')
                    
                    # Buscar dados ITR (trimestrais)
                    st.write("📆 Baixando dados trimestrais (ITR)...")
                    anos_itr = list(range(ano_atual - anos_trimestral, ano_atual + 1))
                    dados_itr = baixar_e_processar_em_paralelo(anos_itr, codigo_cvm, 'ITR')
                    
                    st.write("🔄 Formatando demonstrativos...")
                    
                    dataframes_to_export = {}
                    tipos_processar = []
                    
                    if tipo_dados == "Consolidado":
                        tipos_processar = ['con']
                    elif tipo_dados == "Individual":
                        tipos_processar = ['ind']
                    else:
                        tipos_processar = ['con', 'ind']
                    
                    for tipo in tipos_processar:
                        # Processa DFP
                        bp_dfp_raw = pd.concat([
                            dados_dfp['BPA'][tipo], 
                            dados_dfp['BPP'][tipo]
                        ], ignore_index=True)
                        
                        dre_dfp = formatar_demonstrativo(dados_dfp['DRE'][tipo], 'DRE', is_trimestral=False)
                        bp_dfp = formatar_demonstrativo(bp_dfp_raw, 'BP', is_trimestral=False)
                        dfc_dfp = formatar_demonstrativo(dados_dfp['DFC_MI'][tipo], 'DFC', is_trimestral=False)
                        
                        # Processa ITR
                        bp_itr_raw = pd.concat([
                            dados_itr['BPA'][tipo], 
                            dados_itr['BPP'][tipo]
                        ], ignore_index=True)
                        
                        dre_itr = formatar_demonstrativo(dados_itr['DRE'][tipo], 'DRE', is_trimestral=True)
                        bp_itr = formatar_demonstrativo(bp_itr_raw, 'BP', is_trimestral=True)
                        dfc_itr = formatar_demonstrativo(dados_itr['DFC_MI'][tipo], 'DFC', is_trimestral=True)
                        
                        # Combina dados (com LTM para DRE e DFC)
                        sufixo = "_con" if tipo == 'con' else "_ind"
                        dataframes_to_export[f'DRE{sufixo}'] = combinar_dados(dre_dfp, dre_itr, aplicar_ltm=True)
                        dataframes_to_export[f'BP{sufixo}'] = combinar_dados(bp_dfp, bp_itr, aplicar_ltm=False)
                        dataframes_to_export[f'DFC{sufixo}'] = combinar_dados(dfc_dfp, dfc_itr, aplicar_ltm=True)
                    
                    # Verifica se há dados
                    if not any(not df.empty for df in dataframes_to_export.values()):
                        status.update(label="❌ Nenhum dado encontrado", state="error")
                        st.error("Nenhum dado financeiro encontrado para esta empresa nos anos selecionados.")
                        return
                    
                    st.write("📝 Gerando arquivo Excel...")
                    
                    # Gera arquivo Excel em memória
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        for sheet_name, df in dataframes_to_export.items():
                            if not df.empty:
                                df.to_excel(writer, sheet_name=sheet_name, index=False)
                        
                        # Aba de informações
                        info_data = {
                            'Informação': [
                                'Empresa Analisada',
                                'Código CVM',
                                'Fonte de Dados',
                                'Anos DFP',
                                'Anos ITR',
                                'Data da Extração'
                            ],
                            'Valor': [
                                empresa_selecionada,
                                codigo_cvm,
                                URL_BASE_CVM,
                                f"{ano_atual - anos_historico} a {ano_atual}",
                                f"{ano_atual - anos_trimestral} a {ano_atual}",
                                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            ]
                        }
                        pd.DataFrame(info_data).to_excel(writer, sheet_name='Info', index=False)
                    
                    output.seek(0)
                    
                    status.update(label="✅ Dados processados com sucesso!", state="complete")
                    
                    # Nome do arquivo
                    nome_arquivo = f"DFP_{empresa_selecionada.replace(' ', '_').replace('/', '')}_{datetime.now().strftime('%Y%m%d')}.xlsx"
                    
                    # Botão de download
                    st.download_button(
                        label="⬇️ Baixar Arquivo Excel",
                        data=output,
                        file_name=nome_arquivo,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary",
                        use_container_width=True
                    )
                    
                    # Preview dos dados
                    st.subheader("📋 Preview dos Dados")
                    
                    tabs = st.tabs(list(dataframes_to_export.keys()))
                    for tab, (nome, df) in zip(tabs, dataframes_to_export.items()):
                        with tab:
                            if not df.empty:
                                st.dataframe(df.head(20), use_container_width=True)
                                st.caption(f"Exibindo 20 primeiras linhas de {len(df)} registros.")
                            else:
                                st.info("Sem dados disponíveis.")
                    
                except Exception as e:
                    status.update(label="❌ Erro no processamento", state="error")
                    st.error(f"Erro ao processar dados: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    else:
        st.info("🔎 Busque e selecione uma empresa para exportar os dados financeiros.")
    
    # --- Rodapé informativo ---
    with st.expander("ℹ️ Sobre os Demonstrativos"):
        st.markdown("""
        **Tipos de Demonstrativos:**
        - **DRE (Demonstração do Resultado):** Receitas, custos e lucros.
        - **BP (Balanço Patrimonial):** Ativos, passivos e patrimônio líquido.
        - **DFC (Demonstração de Fluxo de Caixa):** Movimentação de caixa.
        
        **Sufixos:**
        - `_con`: Dados consolidados (inclui subsidiárias)
        - `_ind`: Dados individuais (apenas empresa matriz)
        
        **LTM (Last Twelve Months):**
        - Colunas LTM são calculadas automaticamente para DRE e DFC
        - Fórmula: Ano anterior - trimestres do ano anterior + trimestres do ano atual
        
        **Fonte:** [CVM - Dados Abertos](https://dados.cvm.gov.br/)
        """)
