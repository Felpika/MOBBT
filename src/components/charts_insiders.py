
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

def gerar_grafico_historico_insider(df_historico, ticker):
    """
    Gera um gráfico de barras Plotly para o histórico de volume líquido de insiders.
    """
    if df_historico.empty:
        return go.Figure().update_layout(
            title_text=f"Não há dados de movimentação 'Compra à vista' ou 'Venda à vista' para {ticker}.",
            template="brokeberg", 
            title_x=0.5
        )

    # Adiciona uma coluna de cor para o gráfico (Verde para Compra, Vermelho para Venda)
    df_historico['Cor'] = np.where(df_historico['Volume_Net'] > 0, '#4CAF50', '#F44336')

    fig = px.bar(
        df_historico,
        x='Data',
        y='Volume_Net',
        title=f'Histórico de Volume Líquido Mensal de Insiders: {ticker.upper()}',
        template='brokeberg'
    )

    # Aplica as cores customizadas
    fig.update_traces(marker_color=df_historico['Cor'])

    fig.update_layout(
        title_x=0,
        yaxis_title='Volume Líquido (R$)',
        xaxis_title='Mês',
        showlegend=False
    )
    # Formata o eixo Y para Reais (ex: R$ 1.000.000)
    fig.update_yaxes(tickformat="$,.0f") 
    return fig
