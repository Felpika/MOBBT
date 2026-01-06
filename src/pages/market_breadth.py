
import streamlit as st
import pandas as pd
import yfinance as yf
from scipy import stats
import numpy as np
from src.data_loaders.amplitude import obter_tickers_cvm_amplitude, obter_precos_historicos_amplitude
from src.models.amplitude import calcular_indicadores_amplitude, analisar_retornos_por_faixa
from src.models.indices import get_sector_indices_chart 
from src.components.charts_amplitude import (
    gerar_grafico_historico_amplitude, 
    gerar_histograma_amplitude, 
    gerar_heatmap_amplitude,
    gerar_grafico_amplitude_mm_stacked,
    gerar_grafico_net_highs_lows,
    gerar_grafico_cumulative_highs_lows,
    gerar_grafico_mcclellan,
    gerar_grafico_summation,
    gerar_grafico_macd_breadth,
    gerar_grafico_ifr_breadth
)

def render():
    st.header("Análise de Amplitude de Mercado (Market Breadth)")
    st.info(
        "Esta seção analisa a força interna do mercado, avaliando o comportamento de um grande número "
        "de ações em vez de apenas o índice. Indicadores de amplitude podem fornecer sinais "
        "antecipados de mudanças na tendência principal do mercado."
    )
    st.markdown("---")

    ATIVOS_ANALISE = ['BOVA11.SA', 'SMAL11.SA']
    ANOS_HISTORICO = 10
    PERIODOS_RETORNO = {'1 Mês': 21, '3 Meses': 63, '6 Meses': 126, '1 Ano': 252}

    if 'df_indicadores' not in st.session_state or 'df_analise_base' not in st.session_state:
        with st.spinner("Realizando análise de amplitude... Este processo pode ser demorado na primeira vez..."):
            tickers_cvm = obter_tickers_cvm_amplitude()
            if tickers_cvm:
                precos = obter_precos_historicos_amplitude(tickers_cvm, anos_historico=ANOS_HISTORICO)
                df_analise_base_final = pd.DataFrame(index=precos.index).sort_index()
                
                for ativo in ATIVOS_ANALISE:
                    try:
                        dados_ativo = yf.download(ativo, start=precos.index.min(), end=precos.index.max(), auto_adjust=False, progress=False)
                        if not dados_ativo.empty:
                            if 'Adj Close' in dados_ativo.columns: price_series = dados_ativo[['Adj Close']]
                            else: price_series = dados_ativo[['Close']]
                            price_series.columns = ['price']
                            ativo_label = ativo.replace('.SA', '')
                            for nome_periodo, dias in PERIODOS_RETORNO.items():
                                df_analise_base_final[f'retorno_{nome_periodo} ({ativo_label})'] = price_series['price'].pct_change(periods=dias).shift(-dias) * 100
                    except Exception: pass

                if not precos.empty:
                    st.session_state.df_indicadores = calcular_indicadores_amplitude(precos)
                    st.session_state.df_analise_base = df_analise_base_final.dropna(how='all')
                    st.session_state.analise_amplitude_executada = True
            
            # Setor Indices
            st.session_state.fig_sector = get_sector_indices_chart()

    if st.session_state.get('analise_amplitude_executada', False):
        df_indicadores = st.session_state.df_indicadores
        df_analise_base = st.session_state.df_analise_base

        st.subheader("Visão Geral da Amplitude (MM50/200)")
        df_amplitude_mm_plot = df_indicadores[['breadth_red', 'breadth_yellow', 'breadth_green']].dropna()
        st.plotly_chart(gerar_grafico_amplitude_mm_stacked(df_amplitude_mm_plot), use_container_width=True)
        st.markdown("---")

        st.subheader("Índices Setoriais (Desvio da MMA50)")
        if st.session_state.get('fig_sector'):
             st.plotly_chart(st.session_state.fig_sector, use_container_width=True)
        else:
             st.warning("Gráfico de índices setoriais não gerado.")
        st.markdown("---")

        # Seção 1: Market Breadth (MM200)
        st.subheader("Análise de Market Breadth (% de Ações acima da MM200)")
        mb_series = df_indicadores['market_breadth']
        valor_atual_mb = mb_series.iloc[-1]
        media_hist_mb = mb_series.mean()
        df_analise_mb = df_analise_base.join(mb_series).dropna()
        resultados_mb = analisar_retornos_por_faixa(df_analise_mb, 'market_breadth', 10, 0, 100, '%')
        
        passo_mb = 10
        faixa_atual_valor_mb = int(valor_atual_mb // passo_mb) * passo_mb
        faixa_atual_mb = f'{faixa_atual_valor_mb} a {faixa_atual_valor_mb + passo_mb}%'
        
        c1, c2 = st.columns([1, 2])
        c1.metric("Valor Atual", f"{valor_atual_mb:.2f}%")
        c1.metric("Média Histórica", f"{media_hist_mb:.2f}%")
        c2.plotly_chart(gerar_grafico_historico_amplitude(mb_series, "Histórico Market Breadth", valor_atual_mb, media_hist_mb), use_container_width=True)
        
        c1, c2 = st.columns([1, 2])
        c1.plotly_chart(gerar_histograma_amplitude(mb_series, "Distribuição", valor_atual_mb, media_hist_mb), use_container_width=True)
        for ativo in ATIVOS_ANALISE:
             ativo_clean = ativo.replace('.SA', '')
             sufixo = f" ({ativo_clean})"
             cols_ativo = [c for c in resultados_mb['Retorno Médio'].columns if ativo_clean in c]
             if cols_ativo:
                 df_ret = resultados_mb['Retorno Médio'][cols_ativo].rename(columns=lambda x: x.replace(sufixo, ''))
                 c2.plotly_chart(gerar_heatmap_amplitude(df_ret, faixa_atual_mb, f"Retorno Médio {ativo_clean}"), use_container_width=True)

        st.markdown("---")
        
        # Seção 2: McClellan
        st.subheader("Oscilador McClellan e Summation Index")
        c1, c2 = st.columns(2)
        c1.plotly_chart(gerar_grafico_mcclellan(df_indicadores), use_container_width=True)
        c2.plotly_chart(gerar_grafico_summation(df_indicadores), use_container_width=True)
        st.markdown("---")
        
        # Seção 3: Net Highs/Lows
        st.subheader("Novas Máximas vs Mínimas (52 Semanas)")
        c1, c2 = st.columns(2)
        c1.plotly_chart(gerar_grafico_net_highs_lows(df_indicadores), use_container_width=True)
        c2.plotly_chart(gerar_grafico_cumulative_highs_lows(df_indicadores), use_container_width=True)
        st.markdown("---")

        # Seção 4: IFR e MACD Breadth (Recuperados)
        st.subheader("Momentum de Amplitude (IFR & MACD Breadth)")
        c1, c2 = st.columns(2)
        c1.plotly_chart(gerar_grafico_ifr_breadth(df_indicadores), use_container_width=True)
        c2.plotly_chart(gerar_grafico_macd_breadth(df_indicadores), use_container_width=True)
