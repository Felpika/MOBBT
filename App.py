
import streamlit as st
from streamlit_option_menu import option_menu
from datetime import datetime
from src.config import configurar_tema_brokeberg

# Configuração da Página deve ser a PRIMEIRA coisa
st.set_page_config(layout="wide", page_title="Brokeberg Terminal")

# Configurar Tema
configurar_tema_brokeberg()

# Importar Páginas
from src.pages import (
    dashboard_tesouro,
    credito_privado,
    market_breadth,
    economicos_br,
    dashboard_commodities,
    dashboard_internacional,
    acoes_br,
    radar_insiders,
    calculadora_put,
    screener_put,
    exportador_dfp
)

# --- Sidebar Nuvegação ---
with st.sidebar:
    st.title("Brokeberg Terminal")
    st.caption(f"Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    
    pagina_selecionada = option_menu(
        menu_title="Monitoramento",
        options=[
            "Início",
            "Juros Brasil",
            "Crédito Privado",
            "Amplitude", 
            "Econômicos BR",
            "Commodities",
            "Internacional",
            "Ações BR",
            "Radar de Insiders",
            "Calculadora Put",
            "Screener Put",
            "Exportador DFP",
        ],
        icons=[
            "house",
            "graph-up-arrow",
            "wallet2",
            "water", 
            "bar-chart-line-fill",
            "box-seam",
            "globe-americas",
            "kanban-fill",
            "person-check-fill",
            "calculator",
            "search",
            "file-earmark-spreadsheet",
        ],
        menu_icon="speedometer2",
        default_index=0,
        styles={
            "container": {"padding": "5px !important", "background-color": "transparent"},
            "icon": {"color": "#636EFA", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#262830"},
            "nav-link-selected": {"background-color": "#333952"},
        }
    )

# --- Roteamento ---

if pagina_selecionada == "Início":
    st.header("Bem-vindo ao Brokeberg Terminal")
    st.markdown("---")
    st.info("Selecione uma ferramenta no menu lateral para começar a análise.")
    st.markdown("""
    ### Ferramentas Disponíveis:
    * **Juros Brasil:** Curvas de juros reais, nominais e spreads.
    * **Crédito Privado:** Spreads de debêntures (IDEX JGP).
    * **Amplitude:** Análise de Market Breadth (MM200, IFR, Highs/Lows).
    * **Econômicos BR:** Indicadores macroeconômicos do BCB.
    * **Commodities:** Painel de preços de commodities.
    * **Internacional:** Indicadores do FRED (EUA).
    * **Ações BR:** Análise de Ratio (Long & Short).
    * **Radar de Insiders:** Monitoramento CVM.
    * **Calculadora Put:** Estruturação de venda de opções.
    * **Exportador DFP:** Exporte demonstrativos financeiros de empresas listadas.
    """)

elif pagina_selecionada == "Juros Brasil":
    dashboard_tesouro.render()

elif pagina_selecionada == "Crédito Privado":
    credito_privado.render()

elif pagina_selecionada == "Amplitude":
    market_breadth.render()

elif pagina_selecionada == "Econômicos BR":
    economicos_br.render()

elif pagina_selecionada == "Commodities":
    dashboard_commodities.render()

elif pagina_selecionada == "Internacional":
    dashboard_internacional.render()

elif pagina_selecionada == "Ações BR":
    acoes_br.render()

elif pagina_selecionada == "Radar de Insiders":
    radar_insiders.render()

elif pagina_selecionada == "Calculadora Put":
    calculadora_put.render()

elif pagina_selecionada == "Screener Put":
    screener_put.render()

elif pagina_selecionada == "Exportador DFP":
    exportador_dfp.render()
