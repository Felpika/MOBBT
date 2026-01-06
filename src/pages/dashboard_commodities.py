
import streamlit as st
from src.data_loaders.commodities import carregar_dados_commodities, calcular_variacao_commodities
from src.components.charts import gerar_dashboard_commodities

def render():
    st.title("Monitor de Commodities")
    st.markdown("---")
    
    # Carregar dados
    dados_por_categoria = carregar_dados_commodities()
    if not dados_por_categoria:
        st.error("Falha ao carregar dados de commodities.")
        return

    # Calcular Variações para Tabela
    df_var = calcular_variacao_commodities(dados_por_categoria)
    
    # Exibir Tabela Resumo
    if not df_var.empty:
        st.subheader("Variação de Preços")
        
        # Colorir tabela
        def make_pretty(styler):
            styler.format(subset=[c for c in df_var.columns if 'Variação' in c], formatter="{:.2%}")
            styler.format(subset=['Preço Atual'], formatter="{:.2f}")
            # Aplica cor nas colunas de variação
            for col in [c for c in df_var.columns if 'Variação' in c]:
                 styler.map(lambda v: f"color: {'#4CAF50' if v > 0 else '#F44336'}" if pd.notnull(v) and v!=0 else "", subset=[col])
            return styler

        import pandas as pd # Import needed for styling context if used directly
        # But Streamlit dataframe doesn't support styling via pandas Styler object directly in st.dataframe with full features always.
        # Simple dataframe display with basic formatting
        
        st.dataframe(
            df_var.style.format(formatter={c: "{:.2%}" for c in df_var.columns if 'Variação' in c})
                        .format(formatter={'Preço Atual': "{:.2f}"})
                        .map(lambda v: f"color: {'#39E58C' if v > 0 else '#FF4B4B'}", subset=[c for c in df_var.columns if 'Variação' in c]),
            use_container_width=True,
            height=300
        )
    
    st.markdown("---")
    
    # Exibir Gráficos Sparklines / Históricos
    st.subheader("Gráficos Históricos")
    fig_commodities = gerar_dashboard_commodities(dados_por_categoria)
    st.plotly_chart(fig_commodities, use_container_width=True)
