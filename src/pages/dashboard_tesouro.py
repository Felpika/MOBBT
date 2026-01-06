
import streamlit as st
import pandas as pd
from datetime import datetime
from src.data_loaders.tesouro import obter_dados_tesouro
from src.models.math_utils import (
    calcular_juro_10a_br, 
    calcular_inflacao_implicita, 
    calcular_variacao_curva, 
    calcular_breakeven_historico
)
from src.components.charts import (
    gerar_grafico_historico_tesouro,
    gerar_grafico_ntnb_multiplos_vencimentos,
    gerar_heatmap_variacao_curva,
    gerar_grafico_breakeven_historico,
    gerar_grafico_curva_juros_real_ntnb,
    gerar_grafico_spread_juros,
    gerar_grafico_ettj_generico
)

def render():
    st.title("Análise de Juros e Inflação (Tesouro Direto)")
    st.markdown("---")

    df_tesouro = obter_dados_tesouro()

    if df_tesouro.empty:
        st.error("Não foi possível carregar os dados do Tesouro Direto.")
        return

    # --- Juros Futuros (Proxy 10 anos) ---
    st.subheader("1. Juros Reais de Longo Prazo (Proxy 10 anos)")
    st.caption("Baseado na taxa de compra da NTN-B Principal mais próxima de 10 anos de vencimento.")
    
    serie_10y = calcular_juro_10a_br(df_tesouro)
    if not serie_10y.empty:
        ultimo_juro = serie_10y.iloc[-1]
        delta_juro = ultimo_juro - serie_10y.iloc[-2] if len(serie_10y) > 1 else 0
        
        col1, col2 = st.columns([1, 4])
        col1.metric("Juro Real 10y (IPCA+)", f"{ultimo_juro:.2f}%", f"{delta_juro:.2f}%")
        
        # O gráfico histórico já é gerado por uma função específica ou podemos usar um genérico
        # No código original, ele gerava um gráfico específico. Vamos adaptar.
        # Mas espere, serie_10y é uma Series de taxas.
        # Precisamos de um gráfico disso. Vou usar o index como x.
        import plotly.express as px
        fig_10y = px.line(x=serie_10y.index, y=serie_10y.values, title="Histórico Juro Real 10y (Proxy NTN-B)", template='brokeberg')
        fig_10y.update_layout(yaxis_title="Taxa (% a.a.)", title_x=0)
        col2.plotly_chart(fig_10y, use_container_width=True)
    else:
        st.warning("Dados insuficientes para calcular Proxy 10y.")

    st.markdown("---")

    # --- Análise de Títulos Específicos (Múltiplos Vencimentos) ---
    st.subheader("2. Histórico de Taxas por Vencimento")
    
    tipos_disponiveis = df_tesouro['Tipo Titulo'].dropna().unique().tolist()
    tipo_selecionado = st.selectbox("Selecione o Tipo de Título", tipos_disponiveis, index=tipos_disponiveis.index('Tesouro IPCA+') if 'Tesouro IPCA+' in tipos_disponiveis else 0)
    
    df_tipo = df_tesouro[df_tesouro['Tipo Titulo'] == tipo_selecionado]
    vencimentos_disponiveis = sorted(df_tipo['Data Vencimento'].unique())
    
    container_multiselect = st.container()
    vencimentos_selecionados = container_multiselect.multiselect(
        "Selecione os Vencimentos", 
        vencimentos_disponiveis,
        default=[vencimentos_disponiveis[-1]] if vencimentos_disponiveis else [],
        format_func=lambda x: x.strftime('%d/%m/%Y')
    )
    
    metrica = st.radio("Métrica", ['Taxa Compra Manha', 'PU Compra Manha'], horizontal=True)
    
    fig_multiplo = gerar_grafico_ntnb_multiplos_vencimentos(df_tipo, vencimentos_selecionados, metrica)
    st.plotly_chart(fig_multiplo, use_container_width=True)

    st.markdown("---")

    # --- Curva de Juros e Inflação Implícita ---
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("3. Curva de Juros Real (NTN-B)")
        fig_real = gerar_grafico_curva_juros_real_ntnb(df_tesouro)
        st.plotly_chart(fig_real, use_container_width=True)
        
    with col_b:
        st.subheader("4. Inflação Implícita (Breakeven)")
        df_breakeven = calcular_inflacao_implicita(df_tesouro)
        if not df_breakeven.empty:
            # Plota a curva atual
            import plotly.graph_objects as go
            fig_be = go.Figure()
            fig_be.add_trace(go.Scatter(
                x=df_breakeven['Anos até Vencimento'], 
                y=df_breakeven['Inflação Implícita (% a.a.)'],
                mode='lines+markers',
                name='Implícita',
                line=dict(color='#FFB302')
            ))
            fig_be.update_layout(title="Estrutura a Termo da Inflação Implícita", template='brokeberg', xaxis_title="Anos", yaxis_title="Inflação (%)")
            st.plotly_chart(fig_be, use_container_width=True)
        else:
            st.info("Não foi possível calcular a curva de inflação implícita.")

    # --- Histórico de Breakeven ---
    st.subheader("5. Histórico da Inflação Implícita (5y e 10y)")
    with st.spinner("Calculando histórico de breakeven... (isso pode levar alguns segundos)"):
        # Cache this if slow
        df_be_hist = calcular_breakeven_historico(df_tesouro)
        fig_be_hist = gerar_grafico_breakeven_historico(df_be_hist)
        st.plotly_chart(fig_be_hist, use_container_width=True)

    st.markdown("---")

    # --- Spread de Juros e Heatmap ---
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Spread NTN-F 10y vs 2y")
        fig_spread = gerar_grafico_spread_juros(df_tesouro)
        st.plotly_chart(fig_spread, use_container_width=True)

    with c2:
        st.subheader("Heatmap da Curva Prefixada")
        st.caption("Variação (bps) nos últimos 5 dias")
        df_diff = calcular_variacao_curva(df_tesouro)
        fig_heat = gerar_heatmap_variacao_curva(df_diff)
        st.plotly_chart(fig_heat, use_container_width=True)
        
    st.markdown("---")
    
    # --- ETTJ Curto e Longo Prazo ---
    st.subheader("Dinâmica da Curva Prefixada (ETTJ)")
    t1, t2 = st.tabs(["Curto Prazo (Dias)", "Longo Prazo (Meses)"])
    
    with t1:
        fig_curto = gerar_grafico_ettj_generico(df_tesouro, 'Tesouro Prefixado', 'Curva Prefixada - Curto Prazo')
        st.plotly_chart(fig_curto, use_container_width=True)
        
    with t2:
        fig_longo = gerar_grafico_ettj_generico(df_tesouro, 'Tesouro Prefixado', 'Curva Prefixada - Histórico')
        st.plotly_chart(fig_longo, use_container_width=True)
