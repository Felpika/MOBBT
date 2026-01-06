
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
                 st.markdown(f"**{ativo}**")
                 cols_ativo = [c for c in resultados_mb['Retorno Médio'].columns if ativo_clean in c]
                 if cols_ativo:
                     df_ret = resultados_mb['Retorno Médio'][cols_ativo].rename(columns=lambda x: x.replace(sufixo, ''))
                     df_hit = resultados_mb['Taxa de Acerto'][cols_ativo].rename(columns=lambda x: x.replace(sufixo, ''))
                     
                     ca, cb = st.columns(2)
                     ca.plotly_chart(gerar_heatmap_amplitude(df_ret, faixa_atual_mb, f"Retorno Médio"), use_container_width=True)
                     cb.plotly_chart(gerar_heatmap_amplitude(df_hit, faixa_atual_mb, f"Taxa de Acerto"), use_container_width=True)

        st.markdown("---")

        # --- SEÇÃO 2: MÉDIA GERAL DO IFR ---
        st.subheader("Análise da Média Geral do IFR")
        ifr_media_series = df_indicadores['IFR_media_geral']
        if not ifr_media_series.empty:
            cutoff_ifr = ifr_media_series.index.max() - pd.DateOffset(years=5)
            ifr_media_series = ifr_media_series[ifr_media_series.index >= cutoff_ifr]

        valor_atual_ifr_media = ifr_media_series.iloc[-1]
        media_hist_ifr_media = ifr_media_series.mean()
        df_analise_ifr_media = df_analise_base.join(ifr_media_series).dropna()
        resultados_ifr_media = analisar_retornos_por_faixa(df_analise_ifr_media, 'IFR_media_geral', 5, 0, 100, '')
        passo_ifr_media = 5
        faixa_atual_valor_ifr_media = int(valor_atual_ifr_media // passo_ifr_media) * passo_ifr_media
        faixa_atual_ifr_media = f'{faixa_atual_valor_ifr_media} a {faixa_atual_valor_ifr_media + passo_ifr_media}'

        col1, col2 = st.columns([1,2])
        with col1:
            st.metric("Valor Atual", f"{valor_atual_ifr_media:.2f}")
            st.metric("Média Histórica", f"{media_hist_ifr_media:.2f}")
            z_score_ifr_media = (valor_atual_ifr_media - media_hist_ifr_media) / ifr_media_series.std()
            st.metric("Z-Score (Desvios Padrão)", f"{z_score_ifr_media:.2f}")
            percentil_ifr_media = stats.percentileofscore(ifr_media_series, valor_atual_ifr_media)
            st.metric("Percentil Histórico", f"{percentil_ifr_media:.2f}%")
        with col2:
            st.plotly_chart(gerar_grafico_historico_amplitude(ifr_media_series, "Histórico da Média Geral do IFR (5 Anos)", valor_atual_ifr_media, media_hist_ifr_media), use_container_width=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.plotly_chart(gerar_histograma_amplitude(ifr_media_series, "Distribuição Histórica da Média do IFR", valor_atual_ifr_media, media_hist_ifr_media), use_container_width=True)
        with col2:
             for ativo in ATIVOS_ANALISE:
                 ativo_clean = ativo.replace('.SA', '')
                 sufixo = f" ({ativo_clean})"
                 st.markdown(f"**{ativo}**")
                 cols_ativo = [c for c in resultados_ifr_media['Retorno Médio'].columns if ativo_clean in c]
                 
                 if cols_ativo:
                     df_ret = resultados_ifr_media['Retorno Médio'][cols_ativo].rename(columns=lambda x: x.replace(sufixo, ''))
                     df_hit = resultados_ifr_media['Taxa de Acerto'][cols_ativo].rename(columns=lambda x: x.replace(sufixo, ''))
                     
                     c1, c2 = st.columns(2)
                     c1.plotly_chart(gerar_heatmap_amplitude(df_ret, faixa_atual_ifr_media, "Retorno Médio"), use_container_width=True)
                     c2.plotly_chart(gerar_heatmap_amplitude(df_hit, faixa_atual_ifr_media, "Taxa de Acerto"), use_container_width=True)
        
        st.markdown("---")

        # --- SEÇÃO 3: ANÁLISE DE NET IFR ---
        st.subheader("Análise de Net IFR (% Sobrecomprados - % Sobrevendidos)")
        st.info("O **Net IFR** mede a diferença percentual entre ações sobrecompradas (IFR > 70) e ações sobrevendidas (IFR < 30). Valores positivos indicam euforia, negativos indicam pânico.")
        net_ifr_series = df_indicadores['IFR_net']
        if not net_ifr_series.empty:
            cutoff_net_ifr = net_ifr_series.index.max() - pd.DateOffset(years=5)
            net_ifr_series = net_ifr_series[net_ifr_series.index >= cutoff_net_ifr]

        valor_atual_net_ifr = net_ifr_series.iloc[-1]
        media_hist_net_ifr = net_ifr_series.mean()
        df_analise_net_ifr = df_analise_base.join(net_ifr_series).dropna()
        resultados_net_ifr = analisar_retornos_por_faixa(df_analise_net_ifr, 'IFR_net', 20, -100, 100, '%')

        passo_net_ifr = 20
        # Check to avoid division by zero or invalid invalid integer conversion if series is NaN
        if not np.isnan(valor_atual_net_ifr):
            faixa_atual_valor_net_ifr = int(valor_atual_net_ifr // passo_net_ifr) * passo_net_ifr
            faixa_atual_net_ifr = f'{faixa_atual_valor_net_ifr} a {faixa_atual_valor_net_ifr + passo_net_ifr}%'
        else:
            faixa_atual_net_ifr = "N/A"
        
        col1, col2 = st.columns([1,2])
        with col1:
            st.metric("Valor Atual", f"{valor_atual_net_ifr:.2f}%")
            st.metric("Média Histórica", f"{media_hist_net_ifr:.2f}%")
            z_score_net_ifr = (valor_atual_net_ifr - media_hist_net_ifr) / net_ifr_series.std()
            st.metric("Z-Score (Desvios Padrão)", f"{z_score_net_ifr:.2f}")
            percentil_net_ifr = stats.percentileofscore(net_ifr_series, valor_atual_net_ifr)
            st.metric("Percentil Histórico", f"{percentil_net_ifr:.2f}%")
        with col2:
            st.plotly_chart(gerar_grafico_historico_amplitude(net_ifr_series, "Histórico do Net IFR (5 Anos)", valor_atual_net_ifr, media_hist_net_ifr), use_container_width=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.plotly_chart(gerar_histograma_amplitude(net_ifr_series, "Distribuição Histórica do Net IFR", valor_atual_net_ifr, media_hist_net_ifr, nbins=100), use_container_width=True)
        with col2:
             for ativo in ATIVOS_ANALISE:
                 ativo_clean = ativo.replace('.SA', '')
                 sufixo = f" ({ativo_clean})"
                 st.markdown(f"**{ativo}**")
                 cols_ativo = [c for c in resultados_net_ifr['Retorno Médio'].columns if ativo_clean in c]
                 
                 if cols_ativo:
                     df_ret = resultados_net_ifr['Retorno Médio'][cols_ativo].rename(columns=lambda x: x.replace(sufixo, ''))
                     df_hit = resultados_net_ifr['Taxa de Acerto'][cols_ativo].rename(columns=lambda x: x.replace(sufixo, ''))
                     
                     c1, c2 = st.columns(2)
                     c1.plotly_chart(gerar_heatmap_amplitude(df_ret, faixa_atual_net_ifr, "Retorno Médio"), use_container_width=True)
                     c2.plotly_chart(gerar_heatmap_amplitude(df_hit, faixa_atual_net_ifr, "Taxa de Acerto"), use_container_width=True)

        st.markdown("---")

        # --- SEÇÃO 4: MACD Breadth (SEM HISTOGRAMA/HEATMAP) ---
        st.subheader("MACD Breadth")
        st.info("Mede a porcentagem de ações com tendência de alta (MACD > Sinal). Útil para confirmar a força da tendência do índice.")
        macd_series = df_indicadores['macd_breadth']
        
        if not macd_series.empty:
             cutoff_macd = macd_series.index.max() - pd.DateOffset(years=5)
             macd_series = macd_series[macd_series.index >= cutoff_macd]

        valor_atual_macd = macd_series.iloc[-1]
        media_hist_macd = macd_series.mean()

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
        
        st.markdown("---")

        # --- SEÇÃO 5: Oscilador McClellan e Summation Index (SEM HISTOGRAMA/HEATMAP) ---
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

        col1, col2 = st.columns([1,2])
        with col1:
            st.metric("Valor Atual", f"{valor_atual_mcc:.2f}")
            st.metric("Média Histórica (5A)", f"{media_hist_mcc:.2f}")
            z_score_mcc = (valor_atual_mcc - media_hist_mcc) / mcclellan_series_recent.std()
            st.metric("Z-Score", f"{z_score_mcc:.2f}")
            percentil_mcc = stats.percentileofscore(mcclellan_series_recent, valor_atual_mcc)
            st.metric("Percentil Histórico", f"{percentil_mcc:.2f}%")
        
        with col2:
            c_graph1, c_graph2 = st.columns(2)
            c_graph1.plotly_chart(gerar_grafico_mcclellan(df_indicadores), use_container_width=True)
            c_graph2.plotly_chart(gerar_grafico_summation(df_indicadores), use_container_width=True)
        
        st.markdown("---")
        
        # --- SEÇÃO 6: Novas Máximas vs Mínimas (ADICIONADO HISTOGRAMA/HEATMAP, REMOVIDO ACUMULADO) ---
        st.subheader("Novas Máximas vs Mínimas (52 Semanas)")
        st.info("Saldo líquido de ações atingindo novas máximas de 52 semanas menos novas mínimas. Valores positivos indicam força ampla e tendência de alta.")

        nh_nl_series = df_indicadores['net_highs_lows']
        if not nh_nl_series.empty:
             cutoff_nh = nh_nl_series.index.max() - pd.DateOffset(years=5)
             nh_nl_series_recent = nh_nl_series[nh_nl_series.index >= cutoff_nh]
        else:
             nh_nl_series_recent = nh_nl_series

        valor_atual_nh = nh_nl_series.iloc[-1]
        media_hist_nh = nh_nl_series_recent.mean()
        df_analise_nh = df_analise_base.join(nh_nl_series).dropna()
        
        resultados_nh = analisar_retornos_por_faixa(df_analise_nh, 'net_highs_lows', 20, -200, 200, '')
        passo_nh = 20
        
        if not np.isnan(valor_atual_nh):
            faixa_atual_valor_nh = int(np.floor(valor_atual_nh / passo_nh)) * passo_nh
            faixa_atual_nh = f'{faixa_atual_valor_nh} a {faixa_atual_valor_nh + passo_nh}'
        else:
            faixa_atual_nh = "N/A"

        col1, col2 = st.columns([1, 2])
        with col1:
             st.metric("Saldo Líquido", f"{valor_atual_nh:.0f}")
             st.metric("Média Histórica", f"{media_hist_nh:.0f}")
             z_score_nh = (valor_atual_nh - media_hist_nh) / nh_nl_series_recent.std()
             st.metric("Z-Score", f"{z_score_nh:.2f}")
        with col2:
             st.plotly_chart(gerar_grafico_net_highs_lows(df_indicadores), use_container_width=True)

        col_hist, col_heat = st.columns([1, 2])
        with col_hist:
            st.plotly_chart(gerar_histograma_amplitude(nh_nl_series_recent, "Distribuição (Saldo)", valor_atual_nh, media_hist_nh, nbins=80), use_container_width=True)
        with col_heat:
             for ativo in ATIVOS_ANALISE:
                 ativo_clean = ativo.replace('.SA', '')
                 sufixo = f" ({ativo_clean})"
                 st.markdown(f"**{ativo}**")
                 cols_ativo = [c for c in resultados_nh['Retorno Médio'].columns if ativo_clean in c]
                 
                 if cols_ativo:
                     df_ret = resultados_nh['Retorno Médio'][cols_ativo].rename(columns=lambda x: x.replace(sufixo, ''))
                     df_hit = resultados_nh['Taxa de Acerto'][cols_ativo].rename(columns=lambda x: x.replace(sufixo, ''))
                     
                     c1_hm, c2_hm = st.columns(2)
                     c1_hm.plotly_chart(gerar_heatmap_amplitude(df_ret, faixa_atual_nh, "Retorno Médio"), use_container_width=True)
                     c2_hm.plotly_chart(gerar_heatmap_amplitude(df_hit, faixa_atual_nh, "Taxa de Acerto"), use_container_width=True)

        st.markdown("---")

        # --- SEÇÃO 7: VXEWZ ---
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
