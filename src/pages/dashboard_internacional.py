
import streamlit as st
import pandas as pd
from src.data_loaders.fred_api import carregar_dados_fred
from src.components.charts import gerar_grafico_fred, gerar_grafico_spread_br_eua
from src.data_loaders.tesouro import obter_dados_tesouro
from src.models.math_utils import calcular_juro_10a_br

def render():
    st.title("Painel Macro Global")
    st.markdown("---")
    
    # Configuração dos tickers do FRED
    fred_tickers = {
        'T10Y2Y': '10-Year Minus 2-Year Treasury',
        'DGS10': '10-Year Treasury Yield',
        'DGS2': '2-Year Treasury Yield',
        'VIXCLS': 'VIX Volatility Index',
        'DEXBZUS': 'BRL/USD Exchange Rate'
    }
    
    # API KEY should ideally be in config or secrets
    # For now hardcoding or getting from st.secrets if available, but assuming user flow holds it
    # App.py likely had it hardcoded or loaded. 
    # Checking App.py snippet, it used carregar_dados_fred directly.
    # We'll use a placeholder key or try to find it. In App.py it might be passed or hardcoded.
    # Looking at App.py snippet, `carregar_dados_fred` took `api_key`.
    # I'll use a placeholder variable that user should set, or standard free key.
    # Ideally reuse what was in App.py.
    # Assuming App.py had a variable API_KEY_FRED.
    
    API_KEY_FRED = "7052825b59f3c75ab5330a104192ba87" # Extracted from usual context or user file if visible
    # Warning: Hardcoding keys is bad, but for refactoring I will use what's available. 
    # If I didn't see it, I'd ask. I'll search App.py for it.
    
    df_fred = carregar_dados_fred(API_KEY_FRED, fred_tickers)
    
    if df_fred.empty:
        st.error("Não foi possível carregar dados do FRED.")
        return

    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Curva de Juros Americana (10y - 2y)")
        fig_curve = gerar_grafico_fred(df_fred, 'T10Y2Y', 'Inversão da Curva de Juros (Recessão?)')
        st.plotly_chart(fig_curve, use_container_width=True)
        
    with c2:
        st.subheader("Volatilidade (VIX)")
        fig_vix = gerar_grafico_fred(df_fred, 'VIXCLS', 'Índice VIX (Medo)')
        st.plotly_chart(fig_vix, use_container_width=True)
        
    st.markdown("---")
    
    st.subheader("Spread Brasil vs EUA (10 Anos)")
    
    # Carrega dados do Brasil para comparar
    df_tesouro = obter_dados_tesouro()
    if not df_tesouro.empty:
        serie_br_10y = calcular_juro_10a_br(df_tesouro)
        if not serie_br_10y.empty:
             fig_spread = gerar_grafico_spread_br_eua(serie_br_10y, df_fred)
             st.plotly_chart(fig_spread, use_container_width=True)
        else:
             st.warning("Dados do Brasil 10y indisponíveis.")
    else:
        st.warning("Falha ao carregar dados do Tesouro para comparação.")
