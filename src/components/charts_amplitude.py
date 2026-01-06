
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def gerar_grafico_historico_amplitude(series_dados, titulo, valor_atual, media_hist):
    df_plot = series_dados.to_frame(name='valor').dropna()
    if df_plot.empty: return go.Figure().update_layout(title_text=titulo)
    fig = px.line(df_plot, x=df_plot.index, y='valor', title=titulo, template='brokeberg')
    fig.add_hline(y=media_hist, line_dash="dash", line_color="gray", annotation_text="Média Hist.")
    fig.add_hline(y=valor_atual, line_dash="dot", line_color="yellow", annotation_text=f"Atual: {valor_atual:.2f}")
    
    fig.update_layout(
        showlegend=False, title_x=0, yaxis_title="valor", xaxis_title="Data",
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1A", step="year", stepmode="backward"),
                    dict(count=2, label="2A", step="year", stepmode="backward"),
                    dict(count=5, label="5A", step="year", stepmode="backward"),
                    dict(step="all", label="Tudo")
                ]),
                bgcolor="#333952", font=dict(color="white")
            ),
            type="date"
        )
    )
    if not df_plot.empty:
        end_date = df_plot.index.max()
        start_date = end_date - pd.DateOffset(years=2)
        fig.update_xaxes(range=[start_date, end_date])
        
    return fig

def gerar_histograma_amplitude(series_dados, titulo, valor_atual, media_hist, nbins=50):
    fig = px.histogram(series_dados, title=titulo, nbins=nbins, template='brokeberg')
    fig.add_vline(x=media_hist, line_dash="dash", line_color="gray", annotation_text=f"Média: {media_hist:.2f}")
    fig.add_vline(x=valor_atual, line_dash="dot", line_color="yellow", annotation_text=f"Atual: {valor_atual:.2f}")
    fig.update_layout(showlegend=False, title_x=0)
    return fig

def gerar_heatmap_amplitude(tabela_media, faixa_atual, titulo):
    # Fix: Format text to empty string if NaN, otherwise formatted percentage
    text_values = tabela_media.map(lambda x: f'{x:.1f}%' if pd.notna(x) else '').values
    
    fig = go.Figure(data=go.Heatmap(
        z=tabela_media.values,
        x=[col.replace('retorno_', '') for col in tabela_media.columns],
        y=tabela_media.index,
        hoverongaps=False, colorscale='RdYlGn',
        text=text_values,
        texttemplate="%{text}", showscale=False
    ))
    faixas_y = list(tabela_media.index)
    if faixa_atual in faixas_y:
        y_pos = faixas_y.index(faixa_atual)
        fig.add_shape(type="rect", xref="paper", yref="y", x0=0, y0=y_pos-0.5, x1=1, y1=y_pos+0.5, line=dict(color="White", width=4))
    fig.update_layout(title=titulo, template='brokeberg', yaxis_title='Faixa do Indicador', title_x=0)
    return fig

def gerar_grafico_amplitude_mm_stacked(df_amplitude_plot):
    if df_amplitude_plot.empty:
        return go.Figure().update_layout(title_text="Sem dados para gerar o gráfico.", template='brokeberg')

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_amplitude_plot.index, y=df_amplitude_plot['breadth_green'], 
        name='Acima MM50 e MM200', line=dict(color='#4CAF50', width=1.5),
        fillcolor='rgba(76, 175, 80, 0.4)', fill='tozeroy', mode='lines'
    ))
    
    fig.add_trace(go.Scatter(
        x=df_amplitude_plot.index, y=df_amplitude_plot['breadth_yellow'], 
        name='Abaixo MM50, Acima MM200', line=dict(color='#FFC107', width=1.5),
        fillcolor='rgba(255, 193, 7, 0.4)', fill='tozeroy', mode='lines'
    ))
    
    fig.add_trace(go.Scatter(
        x=df_amplitude_plot.index, y=df_amplitude_plot['breadth_red'], 
        name='Abaixo MM50 e MM200', line=dict(color='#F44336', width=1.5),
        fillcolor='rgba(244, 67, 54, 0.4)', fill='tozeroy', mode='lines'
    ))

    fig.update_layout(
        title_text='Visão Geral: Amplitude de Mercado (MM50/200)',
        title_x=0, template='brokeberg',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title="% Papéis", xaxis_title="Data"
    )
    fig.update_yaxes(range=[0, 100])
    
    # Seletor de Range de Tempo
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1A", step="year", stepmode="backward"),
                dict(count=2, label="2A", step="year", stepmode="backward"),
                dict(count=5, label="5A", step="year", stepmode="backward"),
                dict(step="all", label="Tudo")
            ]),
            bgcolor="#333952", font=dict(color="white")
        ),
        type="date"
    )
    if not df_amplitude_plot.empty:
        end_date = df_amplitude_plot.index.max()
        start_date = end_date - pd.DateOffset(years=2)
        fig.update_xaxes(range=[start_date, end_date])
        
    return fig

def gerar_grafico_net_highs_lows(df_amplitude):
    df_plot = df_amplitude[['net_highs_lows', 'new_highs', 'new_lows']].dropna().copy()
    if df_plot.empty: return go.Figure().update_layout(title_text="Sem dados.")

    fig = go.Figure()
    net = df_plot['net_highs_lows']
    pos = net.where(net >= 0, 0)
    neg = net.where(net < 0, 0)
    
    fig.add_trace(go.Scatter(x=df_plot.index, y=pos, name='Saldo Positivo', line=dict(color='#4CAF50', width=1), fill='tozeroy', fillcolor='rgba(76, 175, 80, 0.5)'))
    fig.add_trace(go.Scatter(x=df_plot.index, y=neg, name='Saldo Negativo', line=dict(color='#F44336', width=1), fill='tozeroy', fillcolor='rgba(244, 67, 54, 0.5)'))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['new_highs'], name='Novas Máximas', line=dict(color='#81C784', dash='dot'), visible='legendonly'))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['new_lows'], name='Novas Mínimas', line=dict(color='#E57373', dash='dot'), visible='legendonly'))

    fig.add_hline(y=0, line_dash="solid", line_color="white", line_width=0.5)
    fig.update_layout(
        title_text='Novas Máximas vs. Novas Mínimas (Saldo Líquido)', 
        title_x=0, template='brokeberg', showlegend=True,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1A", step="year", stepmode="backward"),
                    dict(count=2, label="2A", step="year", stepmode="backward"),
                    dict(count=5, label="5A", step="year", stepmode="backward"),
                    dict(step="all", label="Tudo")
                ]),
                bgcolor="#333952", font=dict(color="white")
            ),
            type="date"
        )
    )
    if not df_plot.empty:
        end_date = df_plot.index.max()
        start_date = end_date - pd.DateOffset(years=2)
        fig.update_xaxes(range=[start_date, end_date])

    return fig

def gerar_grafico_cumulative_highs_lows(df_amplitude):
    """Gera o gráfico acumulado de Net New Highs/Lows."""
    series_cum = df_amplitude['cumulative_net_highs'].dropna()
    
    if series_cum.empty:
        return go.Figure().update_layout(title_text="Sem dados para New Highs/Lows Acumulado", template='brokeberg')
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=series_cum.index,
        y=series_cum,
        name='Net Highs/Lows Acumulado',
        mode='lines',
        line=dict(color='#29B6F6', width=2),
        fill='tozeroy',
        fillcolor='rgba(41, 182, 246, 0.2)'
    ))
    
    fig.update_layout(
        title_text='Acumulado de Novas Máximas - Mínimas (Cumulative AD Line)',
        title_x=0,
        yaxis_title="Acumulado",
        xaxis_title="Data",
        template='brokeberg',
        showlegend=False,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1A", step="year", stepmode="backward"),
                    dict(count=2, label="2A", step="year", stepmode="backward"),
                    dict(count=5, label="5A", step="year", stepmode="backward"),
                    dict(step="all", label="Tudo")
                ]),
                bgcolor="#333952", font=dict(color="white")
            ),
            type="date"
        )
    )
    
    if len(series_cum) > 252*2:
        end_date = series_cum.index.max()
        start_date = end_date - pd.DateOffset(years=2)
        fig.update_xaxes(range=[start_date, end_date])
    
    return fig

def gerar_grafico_mcclellan(df_amplitude):
    s = df_amplitude['mcclellan'].dropna()
    if s.empty: return go.Figure().update_layout(title_text="Sem dados.")

    fig = go.Figure()
    pos = s.where(s >= 0, 0)
    neg = s.where(s < 0, 0)
    
    fig.add_trace(go.Scatter(x=s.index, y=pos, name='Positivo', line=dict(color='#4CAF50', width=1), fill='tozeroy', fillcolor='rgba(76, 175, 80, 0.4)'))
    fig.add_trace(go.Scatter(x=s.index, y=neg, name='Negativo', line=dict(color='#F44336', width=1), fill='tozeroy', fillcolor='rgba(244, 67, 54, 0.4)'))
    fig.add_hline(y=0, line_dash="solid", line_color="white")
    
    fig.update_layout(
        title_text='Oscilador McClellan', title_x=0, template='brokeberg', showlegend=False,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1A", step="year", stepmode="backward"),
                    dict(count=2, label="2A", step="year", stepmode="backward"),
                    dict(count=5, label="5A", step="year", stepmode="backward"),
                    dict(step="all", label="Tudo")
                ]),
                bgcolor="#333952", font=dict(color="white")
            ),
            type="date"
        )
    )
    if not s.empty:
        end_date = s.index.max()
        start_date = end_date - pd.DateOffset(years=2)
        fig.update_xaxes(range=[start_date, end_date])

    return fig

def gerar_grafico_summation(df_amplitude):
    s = df_amplitude['summation_index'].dropna()
    if s.empty: return go.Figure().update_layout(title_text="Sem dados.")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s, name='Summation Index', line=dict(color='#AB47BC', width=2), fill='tozeroy', fillcolor='rgba(171, 71, 188, 0.2)'))
    fig.add_hline(y=0, line_dash="solid", line_color="white")
    
    fig.update_layout(
        title_text='McClellan Summation Index', title_x=0, template='brokeberg',
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1A", step="year", stepmode="backward"),
                    dict(count=2, label="2A", step="year", stepmode="backward"),
                    dict(count=5, label="5A", step="year", stepmode="backward"),
                    dict(step="all", label="Tudo")
                ]),
                bgcolor="#333952", font=dict(color="white")
            ),
            type="date"
        )
    )
    if not s.empty:
        end_date = s.index.max()
        start_date = end_date - pd.DateOffset(years=2)
        fig.update_xaxes(range=[start_date, end_date])
        
    return fig

def gerar_grafico_macd_breadth(df_amplitude):
    """Gera o gráfico de MACD Breadth."""
    s = df_amplitude['macd_breadth'].dropna()
    if s.empty: return go.Figure().update_layout(title_text="Sem dados MACD Breadth.")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s.index, y=s,
        name='% MACD Bullish',
        mode='lines',
        line=dict(color='#00E5FF', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(0, 229, 255, 0.1)'
    ))
    
    fig.update_layout(
        title_text='MACD Breadth (% de Ações com MACD > Sinal)',
        title_x=0, template='brokeberg',
        yaxis_title="%", xaxis_title="Data",
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1A", step="year", stepmode="backward"),
                    dict(count=2, label="2A", step="year", stepmode="backward"),
                    dict(count=5, label="5A", step="year", stepmode="backward"),
                    dict(step="all", label="Tudo")
                ]),
                bgcolor="#333952", font=dict(color="white")
            ),
            type="date"
        )
    )
    fig.update_yaxes(range=[0, 100])
    
    if not s.empty:
        end_date = s.index.max()
        start_date = end_date - pd.DateOffset(years=2)
        fig.update_xaxes(range=[start_date, end_date])
        
    return fig

def gerar_grafico_ifr_breadth(df_amplitude):
    """Gera gráfico de IFR Breadth (Sobrecomprados vs Sobrevendidos)."""
    df_jul = df_amplitude[['IFR_sobrecompradas', 'IFR_sobrevendidas', 'IFR_net']].dropna()
    if df_jul.empty: return go.Figure().update_layout(title_text="Sem dados IFR Breadth.")

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_jul.index, y=df_jul['IFR_sobrecompradas'],
        name='% Sobrecompradas (RSI>70)',
        line=dict(color='#FF5252', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=df_jul.index, y=df_jul['IFR_sobrevendidas'],
        name='% Sobrevendidas (RSI<30)',
        line=dict(color='#69F0AE', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=df_jul.index, y=df_jul['IFR_net'],
        name='Saldo Líquido (SC - SV)',
        line=dict(color='#FFFF00', width=2),
        visible='legendonly'
    ))

    fig.update_layout(
        title_text='IFR Breadth (Sobrecompradas vs Sobrevendidas)',
        title_x=0, template='brokeberg',
        yaxis_title="%", xaxis_title="Data",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1A", step="year", stepmode="backward"),
                    dict(count=2, label="2A", step="year", stepmode="backward"),
                    dict(count=5, label="5A", step="year", stepmode="backward"),
                    dict(step="all", label="Tudo")
                ]),
                bgcolor="#333952", font=dict(color="white")
            ),
            type="date"
        )
    )
    
    if not df_jul.empty:
        end_date = df_jul.index.max()
        start_date = end_date - pd.DateOffset(years=2)
        fig.update_xaxes(range=[start_date, end_date])
        
    return fig
