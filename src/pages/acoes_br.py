
import streamlit as st
import pandas as pd
import yfinance as yf
from src.models.pair_trading import calcular_metricas_ratio, calcular_kpis_ratio
from src.components.charts_pair_trading import gerar_grafico_ratio

@st.cache_data
def carregar_dados_acoes(tickers, period="max"):
    try:
        data = yf.download(tickers, period=period, auto_adjust=True, progress=False)['Close']
        if isinstance(data, pd.Series): 
            data = data.to_frame(tickers[0])
        return data.dropna()
    except Exception:
        return pd.DataFrame()

def render():
    st.header("Ferramentas de Análise de Ações Brasileiras")
    st.markdown("---")
    
    st.subheader("Análise de Ratio de Ativos (Long & Short)")
    st.info("Calcula o ratio entre o preço de dois ativos (Ativo A / Ativo B).")

    col1, col2, col3 = st.columns([0.4, 0.4, 0.2])
    with col1: ticker_a = st.text_input("Ticker do Ativo A (Numerador)", "SMAL11.SA", key="ticker_a_key")
    with col2: ticker_b = st.text_input("Ticker do Ativo B (Denominador)", "BOVA11.SA", key="ticker_b_key")
    with col3: window_size = st.number_input("Janela Móvel (dias)", min_value=20, max_value=500, value=252, key="window_size_key")
    
    if st.button("Analisar Ratio", use_container_width=True):
        st.session_state.analisar_ratio_trigger = True

    if st.session_state.get('analisar_ratio_trigger'):
        with st.spinner(f"Buscando dados..."):
            close_prices = carregar_dados_acoes([ticker_a, ticker_b], period="max")
            
            if close_prices.empty or close_prices.shape[1] < 2:
                st.error("Não foi possível obter dados para ambos os tickers.")
            else:
                ratio_analysis = calcular_metricas_ratio(close_prices, ticker_a, ticker_b, window=window_size)
                fig_ratio = gerar_grafico_ratio(ratio_analysis, ticker_a, ticker_b, window=window_size)
                kpis = calcular_kpis_ratio(ratio_analysis)
                
                if kpis:
                    cols = st.columns(5)
                    cols[0].metric("Ratio Atual", f"{kpis['atual']:.2f}")
                    cols[1].metric("Média Histórica", f"{kpis['media']:.2f}")
                    cols[2].metric("Mínimo", f"{kpis['minimo']:.2f}", f"{kpis['data_minimo'].strftime('%d/%m/%Y')}")
                    cols[3].metric("Máximo", f"{kpis['maximo']:.2f}", f"{kpis['data_maximo'].strftime('%d/%m/%Y')}")
                    cols[4].metric("Var. p/ Média", f"{kpis['variacao_para_media']:.2f}%")
                
                st.plotly_chart(fig_ratio, use_container_width=True)
