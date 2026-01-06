
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from src.data_loaders.idex import carregar_dados_idex, carregar_dados_idex_infra

def gerar_grafico_idex(df_idex):
    if df_idex.empty: return go.Figure().update_layout(title_text="Sem dados IDEX.")
    fig = px.line(df_idex, y=['IDEX Geral (Filtrado)', 'IDEX Low Rated (Filtrado)'], title='Histórico do Spread Médio Ponderado: IDEX JGP', template='brokeberg')
    fig.update_yaxes(tickformat=".2%")
    fig.update_traces(hovertemplate='%{y:.2%}')
    fig.update_layout(title_x=0, yaxis_title='Spread Médio Ponderado (%)', xaxis_title='Data', legend_title_text='Índice', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def gerar_grafico_idex_infra(df_idex_infra):
    if df_idex_infra.empty: return go.Figure().update_layout(title_text="Sem dados IDEX INFRA.")
    fig = px.line(df_idex_infra, y='spread_bps_ntnb', title='Histórico do Spread Médio Ponderado: IDEX INFRA', template='brokeberg')
    fig.update_layout(title_x=0, yaxis_title='Spread Médio (Bps sobre NTNB)', xaxis_title='Data', showlegend=False)
    return fig

def render():
    st.header("IDEX JGP - Indicador de Crédito Privado (Spread/CDI)")
    st.info(
        "O IDEX-CDI mostra o spread médio (prêmio acima do CDI) exigido pelo mercado para comprar debêntures. "
        "Filtramos emissores que passaram por eventos de crédito relevantes."
    )
    df_idex = carregar_dados_idex()
    if not df_idex.empty:
        st.plotly_chart(gerar_grafico_idex(df_idex), use_container_width=True)
    else:
        st.warning("Não foi possível carregar os dados do IDEX-CDI.")

    st.markdown("---")

    st.header("IDEX INFRA - Debêntures de Infraestrutura (Spread/NTN-B)")
    st.info(
        "O IDEX-INFRA mede o spread médio de debêntures incentivadas em relação aos títulos públicos de referência (NTN-Bs)."
    )
    df_idex_infra = carregar_dados_idex_infra()
    if not df_idex_infra.empty:
        st.plotly_chart(gerar_grafico_idex_infra(df_idex_infra), use_container_width=True)
    else:
        st.warning("Não foi possível carregar os dados do IDEX INFRA.")
