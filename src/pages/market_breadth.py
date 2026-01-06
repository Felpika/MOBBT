
import streamlit as st
import pandas as pd
import yfinance as yf
from scipy import stats
import numpy as np
from datetime import datetime, timedelta
from src.data_loaders.amplitude import obter_tickers_cvm_amplitude, obter_precos_historicos_amplitude
from src.models.amplitude import calcular_indicadores_amplitude, analisar_retornos_por_faixa
from src.models.indices import get_sector_indices_chart 
from src.data_loaders.fred_api import carregar_dados_fred
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

        # --- SEÇÃO 1: Market Breadth (MM200) ---
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
        with c1:
            st.metric("Valor Atual", f"{valor_atual_mb:.2f}%")
            st.metric("Média Histórica", f"{media_hist_mb:.2f}%")
            z_score_mb = (valor_atual_mb - media_hist_mb) / mb_series.std()
            st.metric("Z-Score", f"{z_score_mb:.2f}")
            percentil_mb = stats.percentileofscore(mb_series, valor_atual_mb)
            st.metric("Percentil Histórico", f"{percentil_mb:.2f}%")

        with c2:
            st.plotly_chart(gerar_grafico_historico_amplitude(mb_series, "Histórico Market Breadth", valor_atual_mb, media_hist_mb), use_container_width=True)
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.plotly_chart(gerar_histograma_amplitude(mb_series, "Distribuição", valor_atual_mb, media_hist_mb), use_container_width=True)
        with c2:
            for ativo in ATIVOS_ANALISE:
                 ativo_clean = ativo.replace('.SA', '')
                 sufixo = f" ({ativo_clean})"
                 st.markdown(f"**{ativo}**") # Título por ativo
                 cols_ativo = [c for c in resultados_mb['Retorno Médio'].columns if ativo_clean in c]
                 if cols_ativo:
                     df_ret = resultados_mb['Retorno Médio'][cols_ativo].rename(columns=lambda x: x.replace(sufixo, ''))
                     df_hit = resultados_mb['Taxa de Acerto'][cols_ativo].rename(columns=lambda x: x.replace(sufixo, ''))
                     
                     ca, cb = st.columns(2)
                     ca.plotly_chart(gerar_heatmap_amplitude(df_ret, faixa_atual_mb, f"Retorno Médio"), use_container_width=True)
                     cb.plotly_chart(gerar_heatmap_amplitude(df_hit, faixa_atual_mb, f"Taxa de Acerto"), use_container_width=True)

        st.markdown("---")
        
        # --- SEÇÃO 2: MACD Breadth ---
        st.subheader("MACD Breadth")
        st.info("Mede a porcentagem de ações com tendência de alta (MACD > Sinal). Útil para confirmar a força da tendência do índice.")
        macd_series = df_indicadores['macd_breadth']
        
        if not macd_series.empty:
             cutoff_macd = macd_series.index.max() - pd.DateOffset(years=5)
             macd_series = macd_series[macd_series.index >= cutoff_macd]

        valor_atual_macd = macd_series.iloc[-1]
        media_hist_macd = macd_series.mean()
        df_analise_macd = df_analise_base.join(macd_series).dropna()
        resultados_macd = analisar_retornos_por_faixa(df_analise_macd, 'macd_breadth', 10, 0, 100, '%')

        passo_macd = 10
        faixa_atual_valor_macd = int(valor_atual_macd // passo_macd) * passo_macd
        faixa_atual_macd = f'{faixa_atual_valor_macd} a {faixa_atual_valor_macd + passo_macd}%'

        col1, col2 = st.columns([1,2])
        with col1:
            st.metric("Valor Atual (% Bullish)", f"{valor_atual_macd:.2f}%")
            st.metric("Média Histórica", f"{media_hist_macd:.2f}%")
            z_score_macd = (valor_atual_macd - media_hist_macd) / macd_series.std()
            st.metric("Z-Score", f"{z_score_macd:.2f}")
            percentil_macd = stats.percentileofscore(macd_series, valor_atual_macd)
            st.metric("Percentil Histórico", f"{percentil_macd:.2f}%")
        with col2:
            st.plotly_chart(gerar_grafico_historico_amplitude(macd_series, "Histórico MACD Breadth (% Papéis com MACD > Sinal)", valor_atual_macd, media_hist_macd), use_container_width=True)
            
        col1, col2 = st.columns([1, 2])
        with col1:
            st.plotly_chart(gerar_histograma_amplitude(macd_series, "Distribuição Histórica MACD Breadth", valor_atual_macd, media_hist_macd), use_container_width=True)
        with col2:
             for ativo in ATIVOS_ANALISE:
                 ativo_clean = ativo.replace('.SA', '')
                 sufixo = f" ({ativo_clean})"
                 st.markdown(f"**{ativo}**")
                 cols_ativo = [c for c in resultados_macd['Retorno Médio'].columns if ativo_clean in c]
                 
                 if cols_ativo:
                     df_ret = resultados_macd['Retorno Médio'][cols_ativo].rename(columns=lambda x: x.replace(sufixo, ''))
                     df_hit = resultados_macd['Taxa de Acerto'][cols_ativo].rename(columns=lambda x: x.replace(sufixo, ''))
                     
                     c1, c2 = st.columns(2)
                     c1.plotly_chart(gerar_heatmap_amplitude(df_ret, faixa_atual_macd, "Retorno Médio"), use_container_width=True)
                     c2.plotly_chart(gerar_heatmap_amplitude(df_hit, faixa_atual_macd, "Taxa de Acerto"), use_container_width=True)

        st.markdown("---")

        # --- SEÇÃO 3: Oscilador McClellan e Summation Index ---
        st.subheader("Oscilador McClellan e Summation Index")
        st.info("Oscilador McClellan: Momentum de curto prazo. Summation Index: Tendência de médio/longo prazo.")
        
        mcclellan_series = df_indicadores['mcclellan']
        if not mcclellan_series.empty:
             cutoff_mcc = mcclellan_series.index.max() - pd.DateOffset(years=5)
             mcclellan_series_recent = mcclellan_series[mcclellan_series.index >= cutoff_mcc]
        else:
             mcclellan_series_recent = mcclellan_series

        valor_atual_mcc = mcclellan_series.iloc[-1]
        media_hist_mcc = mcclellan_series_recent.mean()

        passo_mcc = 5
        df_analise_mcc = df_analise_base.join(mcclellan_series).dropna()
        resultados_mcc = analisar_retornos_por_faixa(df_analise_mcc, 'mcclellan', passo_mcc, -100, 100, '')
        
        faixa_atual_valor_mcc = int(np.floor(valor_atual_mcc / passo_mcc)) * passo_mcc
        faixa_atual_mcc = f'{faixa_atual_valor_mcc} a {faixa_atual_valor_mcc + passo_mcc}'

        col1, col2 = st.columns([1,2])
        with col1:
            st.metric("Valor Atual", f"{valor_atual_mcc:.2f}")
            st.metric("Média Histórica (5A)", f"{media_hist_mcc:.2f}")
            z_score_mcc = (valor_atual_mcc - media_hist_mcc) / mcclellan_series_recent.std()
            st.metric("Z-Score", f"{z_score_mcc:.2f}")
            percentil_mcc = stats.percentileofscore(mcclellan_series_recent, valor_atual_mcc)
            st.metric("Percentil Histórico", f"{percentil_mcc:.2f}%")
        
        with col2:
            fig_mcclellan = gerar_grafico_mcclellan(df_indicadores)
            st.plotly_chart(fig_mcclellan, use_container_width=True)
            
            st.markdown("#### McClellan Summation Index")
            fig_summation = gerar_grafico_summation(df_indicadores)
            st.plotly_chart(fig_summation, use_container_width=True)

        col_hist, col_heat = st.columns([1, 2])
        with col_hist:
            st.plotly_chart(gerar_histograma_amplitude(mcclellan_series_recent, "Distribuição (McClellan)", valor_atual_mcc, media_hist_mcc, nbins=80), use_container_width=True)
        with col_heat:
             for ativo in ATIVOS_ANALISE:
                 ativo_clean = ativo.replace('.SA', '')
                 sufixo = f" ({ativo_clean})"
                 st.markdown(f"**{ativo}**")
                 cols_ativo = [c for c in resultados_mcc['Retorno Médio'].columns if ativo_clean in c]
                 
                 if cols_ativo:
                     df_ret = resultados_mcc['Retorno Médio'][cols_ativo].rename(columns=lambda x: x.replace(sufixo, ''))
                     df_hit = resultados_mcc['Taxa de Acerto'][cols_ativo].rename(columns=lambda x: x.replace(sufixo, ''))
                     
                     c1, c2 = st.columns(2)
                     c1.plotly_chart(gerar_heatmap_amplitude(df_ret, faixa_atual_mcc, "Retorno Médio"), use_container_width=True)
                     c2.plotly_chart(gerar_heatmap_amplitude(df_hit, faixa_atual_mcc, "Taxa de Acerto"), use_container_width=True)

        st.markdown("---")
        
        # --- SEÇÃO 4: Net Highs/Lows ---
        st.subheader("Novas Máximas vs Mínimas (52 Semanas)")
        c1, c2 = st.columns(2)
        c1.plotly_chart(gerar_grafico_net_highs_lows(df_indicadores), use_container_width=True)
        c2.plotly_chart(gerar_grafico_cumulative_highs_lows(df_indicadores), use_container_width=True)
        st.markdown("---")

        # --- SEÇÃO 5: VXEWZ ---
        st.subheader("Volatilidade Implícita Brasil (VXEWZ)")
        st.info("O índice VXEWZ mede a volatilidade implícita das opções do ETF EWZ. Valores altos indicam stress.")

        FRED_API_KEY = 'd78668ca6fc142a1248f7cb9132916b0'
        df_vxewz = carregar_dados_fred(FRED_API_KEY, {'VXEWZCLS': 'CBOE Brazil ETF Volatility Index (VXEWZ)'})

        if not df_vxewz.empty:
            vxewz_series = df_vxewz['VXEWZCLS'].dropna()
            if not vxewz_series.empty:
                cutoff_vx = vxewz_series.index.max() - pd.DateOffset(years=5)
                vxewz_series_recent = vxewz_series[vxewz_series.index >= cutoff_vx]
            else:
                vxewz_series_recent = vxewz_series

            valor_atual_vx = vxewz_series.iloc[-1]
            media_hist_vx = vxewz_series_recent.mean()

            # Merge com base para retornos
            df_analise_vx = df_analise_base.join(vxewz_series, how='inner').dropna()
            
            passo_vx = 5
            resultados_vx = analisar_retornos_por_faixa(df_analise_vx, 'VXEWZCLS', passo_vx, 10, 100, '') 
            
            faixa_atual_val_vx = int(valor_atual_vx // passo_vx) * passo_vx
            faixa_atual_vx = f'{faixa_atual_val_vx} a {faixa_atual_val_vx + passo_vx}'

            col1, col2 = st.columns([1,2])
            with col1:
                st.metric("Valor Atual", f"{valor_atual_vx:.2f}")
                st.metric("Média Histórica (5A)", f"{media_hist_vx:.2f}")
                z_score_vx = (valor_atual_vx - media_hist_vx) / vxewz_series_recent.std()
                st.metric("Z-Score", f"{z_score_vx:.2f}")
                percentil_vx = stats.percentileofscore(vxewz_series_recent, valor_atual_vx)
                st.metric("Percentil Histórico", f"{percentil_vx:.2f}%")
            
            with col2:
                fig_vxewz = gerar_grafico_historico_amplitude(vxewz_series, "Histórico VXEWZ", valor_atual_vx, media_hist_vx)
                st.plotly_chart(fig_vxewz, use_container_width=True)

            col_hist, col_heat = st.columns([1, 2])
            with col_hist:
                st.plotly_chart(gerar_histograma_amplitude(vxewz_series_recent, "Distribuição (VXEWZ)", valor_atual_vx, media_hist_vx, nbins=50), use_container_width=True)
            with col_heat:
                 for ativo in ATIVOS_ANALISE:
                     sufixo = f" ({ativo})"
                     st.markdown(f"**{ativo}**")
                     cols_ativo = [c for c in resultados_vx['Retorno Médio'].columns if ativo in c]
                     
                     if cols_ativo:
                         df_ret = resultados_vx['Retorno Médio'][cols_ativo].rename(columns=lambda x: x.replace(sufixo, ''))
                         df_hit = resultados_vx['Taxa de Acerto'][cols_ativo].rename(columns=lambda x: x.replace(sufixo, ''))
                         
                         c1, c2 = st.columns(2)
                         c1.plotly_chart(gerar_heatmap_amplitude(df_ret, faixa_atual_vx, "Retorno Médio"), use_container_width=True)
                         c2.plotly_chart(gerar_heatmap_amplitude(df_hit, faixa_atual_vx, "Taxa de Acerto"), use_container_width=True)

        else:
            st.warning("Não foi possível carregar os dados do índice de volatilidade VXEWZ.")
