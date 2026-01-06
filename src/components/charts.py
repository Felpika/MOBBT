
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from plotly.subplots import make_subplots

def gerar_grafico_historico_tesouro(df, tipo, vencimento, metrica='Taxa Compra Manha'):
    """
    Gera gráfico histórico simples para um título específico.
    """
    df_filtrado = df[(df['Tipo Titulo'] == tipo) & (df['Data Vencimento'] == vencimento)].sort_values('Data Base')
    titulo = f'Histórico da Taxa de Compra: {tipo} (Venc. {vencimento.strftime("%d/%m/%Y")})' if metrica == 'Taxa Compra Manha' else f'Histórico do Preço Unitário (PU): {tipo} (Venc. {vencimento.strftime("%d/%m/%Y")})'
    eixo_y = "Taxa de Compra (% a.a.)" if metrica == 'Taxa Compra Manha' else "Preço Unitário (R$)"
    fig = px.line(df_filtrado, x='Data Base', y=metrica, title=titulo, template='brokeberg')
    fig.update_layout(title_x=0, yaxis_title=eixo_y, xaxis_title="Data")
    return fig

def gerar_grafico_ntnb_multiplos_vencimentos(df_ntnb_all, vencimentos, metrica):
    """
    Gera um gráfico comparativo para múltiplos vencimentos de NTN-Bs.
    """
    fig = go.Figure()

    if not vencimentos:
        return fig.update_layout(title_text="Selecione um ou mais vencimentos", template="brokeberg")

    for venc in vencimentos:
        df_venc = df_ntnb_all[df_ntnb_all['Data Vencimento'] == venc].sort_values('Data Base')
        if not df_venc.empty:
            nome_base = df_venc['Tipo Titulo'].iloc[0].replace("Tesouro ", "")
            fig.add_trace(go.Scatter(
                x=df_venc['Data Base'],
                y=df_venc[metrica],
                mode='lines',
                line=dict(shape='spline', smoothing=1.0),
                name=f'{nome_base} {venc.year}'
            ))

    titulo = f'Histórico da Taxa de Compra' if metrica == 'Taxa Compra Manha' else f'Histórico do Preço Unitário (PU)'
    eixo_y = "Taxa de Compra (% a.a.)" if metrica == 'Taxa Compra Manha' else "Preço Unitário (R$)"
    
    fig.update_layout(
        title_text=titulo, title_x=0,
        yaxis_title=eixo_y, xaxis_title="Data",
        template='brokeberg',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1a", step="year", stepmode="backward"),
                dict(count=3, label="3a", step="year", stepmode="backward"),
                dict(count=5, label="5a", step="year", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(step="all", label="Tudo")
            ]),
            bgcolor="#333952", font=dict(color="white")
        )
    )
    
    if not df_ntnb_all.empty:
        end_date = df_ntnb_all['Data Base'].max()
        start_date = end_date - pd.DateOffset(years=5)
        fig.update_xaxes(range=[start_date, end_date])

    return fig

def gerar_heatmap_variacao_curva(df_diff):
    """
    Gera um heatmap de variação diária da curva de juros (Pre).
    """
    if df_diff.empty:
        return go.Figure().update_layout(title_text="Sem dados suficientes", template='brokeberg')

    # Ajusta labels do eixo X
    data_ref = df_diff.index.max()
    x_labels = []
    for col in df_diff.columns:
        anos = (col - data_ref).days / 365.25
        anos_rounded = round(anos * 2) / 2
        x_labels.append(f"{anos_rounded}y")

    y_labels = df_diff.index.strftime('%d/%m')

    fig = go.Figure(data=go.Heatmap(
        z=df_diff.values,
        x=x_labels,
        y=y_labels,
        colorscale='RdYlGn_r', 
        zmid=0,
        text=df_diff.values,
        texttemplate="%{text:+g}",
        textfont={"size": 11},
        hoverongaps=False,
        ygap=2, xgap=2
    ))

    fig.update_layout(
        title='Variação Diária da Curva Prefixada (bps)',
        template='brokeberg',
        title_x=0,
        xaxis_title="Vencimento (Prazo)",
        yaxis_title="Data",
        height=350
    )
    return fig

def gerar_grafico_breakeven_historico(df_breakeven):
    if df_breakeven.empty:
         return go.Figure().update_layout(title_text="Sem dados para histórico de inflação implícita.", template='brokeberg')

    fig = go.Figure()
    cores = {'Breakeven 5y': '#FFA726', 'Breakeven 10y': '#EF5350'}
    
    for col in df_breakeven.columns:
        fig.add_trace(go.Scatter(
            x=df_breakeven.index, 
            y=df_breakeven[col], 
            name=col, 
            mode='lines',
            connectgaps=False, 
            line=dict(color=cores.get(col, '#CCCCCC'), width=2)
        ))

    fig.add_hline(y=3.0, line_dash="dot", line_color="gray", annotation_text="Meta 3%", annotation_position="top left")

    fig.update_layout(
        title='Histórico de Inflação Implícita (Breakeven)',
        template='brokeberg',
        title_x=0,
        xaxis_title="Data",
        yaxis_title="Inflação Implícita (% a.a.)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def gerar_grafico_curva_juros_real_ntnb(df):
    if df.empty or 'Data Base' not in df.columns:
        return go.Figure().update_layout(title_text="Não há dados disponíveis.", template='brokeberg')
    
    tipos_ntnb = ['Tesouro IPCA+', 'Tesouro IPCA+ com Juros Semestrais']
    df_recente = df[df['Data Base'] == df['Data Base'].max()].copy()
    df_ntnb = df_recente[df_recente['Tipo Titulo'].isin(tipos_ntnb)].copy()
    
    if df_ntnb.empty:
        return go.Figure().update_layout(title_text="Não há dados de NTN-Bs disponíveis.", template='brokeberg')
    
    df_ntnb = df_ntnb.sort_values('Tipo Titulo', ascending=False).drop_duplicates('Data Vencimento')
    df_ntnb = df_ntnb.sort_values('Data Vencimento')
    
    data_ref = df_recente['Data Base'].max()
    df_ntnb['Anos até Vencimento'] = ((pd.to_datetime(df_ntnb['Data Vencimento']) - data_ref).dt.days / 365.25)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_ntnb['Anos até Vencimento'],
        y=df_ntnb['Taxa Compra Manha'],
        mode='lines',
        line=dict(color='#4CAF50', width=2.5, shape='spline', smoothing=1.0),
        name='Juros Real (IPCA+)',
        hovertemplate="Vencimento: %{customdata[0]}<br>Prazo: %{x:.1f} anos<br>Taxa Real: %{y:.2f}% a.a.<extra></extra>",
        customdata=np.stack([df_ntnb['Data Vencimento'].dt.strftime('%d/%m/%Y')], axis=-1)
    ))
    
    fig.update_layout(
        title=f'Curva de Juros Real (NTN-Bs) - {data_ref.strftime("%d/%m/%Y")}',
        template='brokeberg',
        title_x=0,
        xaxis_title='Prazo até o Vencimento (anos)',
        yaxis_title='Taxa de Juros Real (% a.a.)',
        showlegend=False
    )
    fig.update_yaxes(tickformat=".2f")
    return fig

def gerar_grafico_spread_juros(df):
    df_ntnf = df[df['Tipo Titulo'] == 'Tesouro Prefixado com Juros Semestrais'].copy()
    if df_ntnf.empty: return go.Figure().update_layout(title_text="Dados insuficientes para spread.")

    data_recente = df_ntnf['Data Base'].max()
    df_dia_recente = df_ntnf[df_ntnf['Data Base'] == data_recente]
    vencimentos_recentes = df_dia_recente['Data Vencimento'].unique()

    if len(vencimentos_recentes) < 2: return go.Figure().update_layout(title_text="Vencimentos insuficientes.")

    target_2y = pd.to_datetime(data_recente) + pd.DateOffset(years=2)
    target_10y = pd.to_datetime(data_recente) + pd.DateOffset(years=10)

    venc_curto_fixo = min(vencimentos_recentes, key=lambda d: abs(d - target_2y))
    venc_longo_fixo = min(vencimentos_recentes, key=lambda d: abs(d - target_10y))

    df_curto = df_ntnf[df_ntnf['Data Vencimento'] == venc_curto_fixo][['Data Base', 'Taxa Compra Manha']]
    df_curto = df_curto.rename(columns={'Taxa Compra Manha': 'Taxa Curta'}).set_index('Data Base')
    
    df_longo = df_ntnf[df_ntnf['Data Vencimento'] == venc_longo_fixo][['Data Base', 'Taxa Compra Manha']]
    df_longo = df_longo.rename(columns={'Taxa Compra Manha': 'Taxa Longa'}).set_index('Data Base')

    df_merged = pd.merge(df_curto, df_longo, on='Data Base', how='inner')
    df_merged['Spread'] = (df_merged['Taxa Longa'] - df_merged['Taxa Curta']) * 100 
    df_spread_final = df_merged[['Spread']].dropna().sort_index()

    if df_spread_final.empty: return go.Figure().update_layout(title_text="Sem dados de spread.")

    df_plot = df_spread_final.reset_index()
    df_plot.columns = ['Data', 'Spread']
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_plot['Data'], y=df_plot['Spread'],
        mode='lines', fill='tozeroy',
        line=dict(color='#636EFA', shape='spline', smoothing=1.0),
        name='Spread'
    ))
    
    end_date = df_spread_final.index.max()
    start_date_real = df_spread_final.index.min()
    
    fig.update_layout(
        title=f'Spread de Juros (Fixo): NTN-F {venc_longo_fixo.strftime("%Y")} vs. NTN-F {venc_curto_fixo.strftime("%Y")}',
        template='brokeberg', title_x=0,
        yaxis_title="Diferença (Basis Points)", xaxis_title="Data",
        showlegend=False
    )
    
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1A", step="year", stepmode="backward"),
                dict(count=2, label="2A", step="year", stepmode="backward"),
                dict(count=5, label="5A", step="year", stepmode="backward"),
                dict(step="all", label="Máx")
            ]),
            bgcolor="#333952", font=dict(color="white")
        )
    )
    start_date_default = max(end_date - pd.DateOffset(years=5), start_date_real)
    fig.update_xaxes(range=[start_date_default, end_date])
    return fig

def gerar_grafico_ettj_generico(df, tipo_titulo, titulo_grafico):
    """
    Função genérica para ETTJ de Curto e Longo Prazo
    """
    df_prefixado = df[df['Tipo Titulo'] == tipo_titulo].copy()
    if df_prefixado.empty: return go.Figure().update_layout(title_text=f"Não há dados para '{tipo_titulo}'.")
    
    datas_disponiveis = sorted(df_prefixado['Data Base'].unique())
    data_recente = datas_disponiveis[-1]
    
    # Define targets baseados no título
    if "Curto Prazo" in titulo_grafico:
        targets = {f'Hoje ({data_recente.strftime("%d/%m/%Y")})': data_recente, 
                   '1 dia Atrás': data_recente - pd.DateOffset(days=1),
                   '1 Semana Atrás': data_recente - pd.DateOffset(weeks=1)}
    else:
        targets = {f'Hoje ({data_recente.strftime("%d/%m/%Y")})': data_recente, 
                   '1 Mês Atrás': data_recente - pd.DateOffset(months=1), 
                   '6 Meses Atrás': data_recente - pd.DateOffset(months=6), 
                   '1 Ano Atrás': data_recente - pd.DateOffset(years=1)}

    fig = go.Figure()
    datas_plotadas = []
    
    for legenda_base, data_alvo in targets.items():
        datas_validas = [d for d in datas_disponiveis if d <= data_alvo]
        if datas_validas:
            data_real = max(datas_validas)
            if data_real not in datas_plotadas:
                datas_plotadas.append(data_real)
                legenda_final = f'{" ".join(legenda_base.split(" ")[:2])} ({data_real.strftime("%d/%m/%Y")})' if not legenda_base.startswith('Hoje') else legenda_base
                
                df_data = df_prefixado[df_prefixado['Data Base'] == data_real].sort_values('Data Vencimento')
                df_data['Dias Uteis'] = np.busday_count(df_data['Data Base'].values.astype('M8[D]'), df_data['Data Vencimento'].values.astype('M8[D]'))
                df_data['Anos até Vencimento'] = df_data['Dias Uteis'] / 252 
                
                line_style = dict(dash='dash', shape='spline', smoothing=1.0) if not legenda_base.startswith('Hoje') else dict(shape='spline', smoothing=1.0)
                fig.add_trace(go.Scatter(x=df_data['Anos até Vencimento'], y=df_data['Taxa Compra Manha'], mode='lines', name=legenda_final, line=line_style))

    fig.update_layout(
        title_text=titulo_grafico, title_x=0, 
        xaxis_title='Prazo até o Vencimento (anos)', 
        yaxis_title='Taxa (% a.a.)', 
        template='brokeberg', 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def gerar_grafico_fred(df, ticker, titulo):
    if ticker not in df.columns or df[ticker].isnull().all():
        return go.Figure().update_layout(title_text=f"Dados para {ticker} não encontrados.")
    fig = px.line(df, y=ticker, title=titulo, template='brokeberg')
    if ticker == 'T10Y2Y':
        fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Inversão", annotation_position="bottom right")
    
    end_date = df.index.max()
    buttons = []
    periods = {'6M': 182, '1A': 365, '2A': 730, '5A': 1825, '10A': 3650, 'Máx': 'max'}
    for label, days in periods.items():
        start_date = df.index.min() if days == 'max' else end_date - timedelta(days=days)
        buttons.append(dict(method='relayout', label=label, args=[{'xaxis.range': [start_date, end_date], 'yaxis.autorange': True}]))
    
    fig.update_layout(
        title_x=0, yaxis_title="Pontos Percentuais (%)", xaxis_title="Data", showlegend=False,
        updatemenus=[dict(type="buttons", direction="right", showactive=True, x=1, xanchor="right", y=1.05, yanchor="bottom", buttons=buttons)]
    )
    
    start_date_1y = end_date - timedelta(days=365)
    filtered_series = df.loc[start_date_1y:end_date, ticker].dropna()
    fig.update_xaxes(range=[start_date_1y, end_date])
    if not filtered_series.empty:
        min_y, max_y = filtered_series.min(), filtered_series.max()
        padding = (max_y - min_y) * 0.10 if (max_y - min_y) > 0 else 0.5
        fig.update_yaxes(range=[min_y - padding, max_y + padding])
    return fig

def gerar_grafico_spread_br_eua(df_br, df_usa):
    # Assume df_br is Series of 10y yield, df_usa is DataFrame with DGS10
    df_br_s = df_br.copy()
    df_br_s.name = 'BR10Y'
    if isinstance(df_usa, pd.DataFrame):
        df_usa_s = df_usa['DGS10']
    else:
        df_usa_s = df_usa
        
    df_merged = pd.merge(df_br_s, df_usa_s, left_index=True, right_index=True, how='inner')
    df_merged['Spread'] = df_merged['BR10Y'] - df_merged['DGS10']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_merged.index, y=df_merged['Spread'],
        mode='lines', line=dict(color='#636EFA', shape='spline', smoothing=1.0),
        name='Spread'
    ))
    
    end_date = df_merged.index.max()
    buttons = []
    periods = {'1A': 365, '2A': 730, '5A': 1825, 'Máx': 'max'}
    for label, days in periods.items():
        start_date = df_merged.index.min() if days == 'max' else end_date - timedelta(days=days)
        buttons.append(dict(method='relayout', label=label, args=[{'xaxis.range': [start_date, end_date], 'yaxis.autorange': True}]))
    
    fig.update_layout(
        title='Spread de Juros 10 Anos: NTN-B (Brasil) vs. Treasury (EUA)',
        template='brokeberg', title_x=0,
        yaxis_title="Diferença (Pontos Percentuais)", xaxis_title="Data", showlegend=False,
        updatemenus=[dict(type="buttons", direction="right", showactive=True, x=1, xanchor="right", y=1.05, yanchor="bottom", buttons=buttons)]
    )
    
    start_date_1y = end_date - timedelta(days=365)
    filtered_series = df_merged.loc[start_date_1y:end_date, 'Spread'].dropna()
    fig.update_xaxes(range=[start_date_1y, end_date])
    if not filtered_series.empty:
        min_y, max_y = filtered_series.min(), filtered_series.max()
        padding = (max_y - min_y) * 0.10 if (max_y - min_y) > 0 else 0.5
        fig.update_yaxes(range=[min_y - padding, max_y + padding])
    return fig

def plot_sector_indices_chart(results, index_meta):
    """
    Plots the calculated sector indices deviations.
    """
    fig = go.Figure()

    for sector in results.columns:
        meta = index_meta.get(sector, {})
        color = meta.get('color', '#808080')
        label = meta.get('name', sector)
        
        series = results[sector].dropna()
        if series.empty:
            continue
            
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series,
            mode='lines',
            name=label,
            line=dict(color=color, width=2),
            hovertemplate=f"<b>{label}</b><br>Data: %{{x|%d/%m/%Y}}<br>Desvio: %{{y:.2f}}%<extra></extra>"
        ))
        
        last_val = series.iloc[-1]
        last_date = series.index[-1]
        
        fig.add_annotation(
            x=last_date, y=last_val,
            text=f"{last_val:.2f}%",
            showarrow=False, xanchor="left",
            font=dict(color=color, size=12),
            bgcolor="#161B22", bordercolor=color, borderwidth=1,
            xshift=5
        )

    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)

    fig.update_layout(
        title="<b>Desvio dos Índices Setoriais vs MMA 50</b>",
        yaxis_title="Variação % da MMA 50",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=600,
        margin=dict(l=40, r=40, t=80, b=40),
        hovermode="x unified",
        template='brokeberg'
    )
    return fig

def colorir_negativo_positivo(val):
    if pd.isna(val) or val == 0: return ''
    return f"color: {'#4CAF50' if val > 0 else '#F44336'}"

def gerar_dashboard_commodities(dados_preco_por_categoria):
    all_commodity_names = [name for df in dados_preco_por_categoria.values() for name in df.columns]
    total_subplots = len(all_commodity_names)
    if total_subplots == 0: return go.Figure().update_layout(title_text="Nenhum dado de commodity disponível.")
    num_cols, num_rows = 4, int(np.ceil(total_subplots / 4))
    fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=all_commodity_names)
    idx = 0
    for df_cat in dados_preco_por_categoria.values():
        for commodity_name in df_cat.columns:
            row, col = (idx // num_cols) + 1, (idx % num_cols) + 1
            fig.add_trace(go.Scatter(x=df_cat.index, y=df_cat[commodity_name], mode='lines', name=commodity_name), row=row, col=col)
            idx += 1
    end_date = datetime.now(); buttons = []; 
    periods = {'1M': 30, '3M': 91, '6M': 182, 'YTD': 'ytd', '1A': 365, '5A': 365*5, '10A': 3650, 'Máx': 'max'}
    for label, days in periods.items():
        if days == 'ytd': start_date = datetime(end_date.year, 1, 1)
        elif days == 'max': start_date = min([df.index.min() for df in dados_preco_por_categoria.values() if not df.empty])
        else: start_date = end_date - timedelta(days=days)
        update_args = {}
        for i in range(1, total_subplots + 1):
            update_args[f'xaxis{i if i > 1 else ""}.range'], update_args[f'yaxis{i if i > 1 else ""}.autorange'] = [start_date, end_date], True
        buttons.append(dict(method='relayout', label=label, args=[update_args]))
    active_button_index = list(periods.keys()).index('1A') if '1A' in list(periods.keys()) else 4
    fig.update_layout(title_text="Dashboard de Preços Históricos de Commodities", title_x=0, template="plotly_dark", height=250 * num_rows, showlegend=False,
                        updatemenus=[dict(type="buttons", direction="right", showactive=True, x=1, xanchor="right", y=1.05, yanchor="bottom", buttons=buttons, active=active_button_index)])
    start_date_1y = end_date - timedelta(days=365); idx = 0
    for df_cat in dados_preco_por_categoria.values():
        for i, commodity_name in enumerate(df_cat.columns, start=idx):
            fig.layout[f'xaxis{i+1 if i+1 > 1 else ""}.range'] = [start_date_1y, end_date]
            series = df_cat[commodity_name]; filtered_series = series[(series.index >= start_date_1y) & (series.index <= end_date)].dropna()
            if not filtered_series.empty:
                min_y, max_y = filtered_series.min(), filtered_series.max(); padding = (max_y - min_y) * 0.05
                fig.layout[f'yaxis{i+1 if i+1 > 1 else ""}.range'] = [min_y - padding, max_y + padding]
            else: fig.layout[f'yaxis{i+1 if i+1 > 1 else ""}.autorange'] = True
        idx += len(df_cat.columns)
    return fig
