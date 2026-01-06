
import streamlit as st
import plotly.express as px
from src.data_loaders.bcb_api import carregar_dados_bcb

def render():
    st.header("Monitor de Indicadores Econômicos Nacionais")
    st.markdown("---")
    st.subheader("Indicadores Macroeconômicos (BCB)")
    
    df_bcb, config_bcb = carregar_dados_bcb()
    
    if not df_bcb.empty:
        col1, col2 = st.columns(2)
        with col1:
             data_inicio = st.date_input("Data de Início", df_bcb.index.min().date(), min_value=df_bcb.index.min().date(), max_value=df_bcb.index.max().date(), key='bcb_start')
        with col2:
             data_fim = st.date_input("Data de Fim", df_bcb.index.max().date(), min_value=data_inicio, max_value=df_bcb.index.max().date(), key='bcb_end')
        
        df_filtrado_bcb = df_bcb.loc[str(data_inicio):str(data_fim)]
        
        num_cols_bcb = 3
        cols_bcb = st.columns(num_cols_bcb)
        
        for i, nome_serie in enumerate(df_filtrado_bcb.columns):
            fig_bcb = px.line(df_filtrado_bcb, x=df_filtrado_bcb.index, y=nome_serie, title=nome_serie, template='brokeberg')
            fig_bcb.update_layout(title_x=0)
            cols_bcb[i % num_cols_bcb].plotly_chart(fig_bcb, use_container_width=True)
    else:
        st.warning("Não foi possível carregar os dados do BCB.")
