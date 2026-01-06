
import pandas as pd
import pandas_ta as ta

def calcular_indicadores_amplitude(_precos_fechamento, rsi_periodo=14):
    """Calcula indicadores de amplitude, incluindo Highs/Lows, McClellan e MACD Breadth."""
    
    # 1. Market Breadth (MM200)
    mma200 = _precos_fechamento.rolling(window=200, min_periods=50).mean()
    acima_da_media = _precos_fechamento > mma200
    percentual_acima_media = (acima_da_media.sum(axis=1) / _precos_fechamento.notna().sum(axis=1)) * 100
    
    # 2. Categorias MM50 vs MM200
    mma50 = _precos_fechamento.rolling(window=50, min_periods=20).mean()
    total_papeis_validos = _precos_fechamento.notna().sum(axis=1)
    
    cat_red = ((_precos_fechamento < mma50) & (_precos_fechamento < mma200)).sum(axis=1) / total_papeis_validos * 100
    cat_yellow = ((_precos_fechamento < mma50) & (_precos_fechamento > mma200)).sum(axis=1) / total_papeis_validos * 100
    cat_green = ((_precos_fechamento > mma50) & (_precos_fechamento > mma200)).sum(axis=1) / total_papeis_validos * 100

    # 3. IFR (usando pandas_ta)
    # Apply applies to columns by default (axis=0)
    ifr_individual = _precos_fechamento.apply(
        lambda x: ta.rsi(x, length=rsi_periodo) if len(x.dropna()) >= rsi_periodo else pd.Series(index=x.index, dtype=float),
        axis=0
    )
    total_valido_ifr = ifr_individual.notna().sum(axis=1)
    sobrecompradas = (ifr_individual > 70).sum(axis=1) / total_valido_ifr * 100
    sobrevendidas = (ifr_individual < 30).sum(axis=1) / total_valido_ifr * 100

    # 4. Novas Máximas e Mínimas (52 Semanas / 252 Dias)
    rolling_max = _precos_fechamento.rolling(window=252, min_periods=200).max()
    rolling_min = _precos_fechamento.rolling(window=252, min_periods=200).min()
    new_highs = (_precos_fechamento >= rolling_max).sum(axis=1)
    new_lows = (_precos_fechamento <= rolling_min).sum(axis=1)
    net_highs_lows = new_highs - new_lows

    # 5. Oscilador McClellan
    diff_precos = _precos_fechamento.diff()
    advances = (diff_precos > 0).sum(axis=1)
    declines = (diff_precos < 0).sum(axis=1)
    net_advances = advances - declines
    ema_19 = net_advances.ewm(span=19, adjust=False).mean()
    ema_39 = net_advances.ewm(span=39, adjust=False).mean()
    mcclellan_osc = ema_19 - ema_39

    # --- 6. MACD Breadth ---
    ema12 = _precos_fechamento.ewm(span=12, adjust=False).mean()
    ema26 = _precos_fechamento.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histograma_macd = macd_line - signal_line
    
    macd_bullish = (histograma_macd > 0).sum(axis=1)
    validos_macd = histograma_macd.notna().sum(axis=1)
    percentual_macd_bullish = (macd_bullish / validos_macd).fillna(0) * 100

    # --- 7. Summation & Cumulative ---
    summation_index = mcclellan_osc.cumsum()
    cumulative_net_highs_lows = net_highs_lows.cumsum()

    df_amplitude = pd.DataFrame({
        'market_breadth': percentual_acima_media,
        'IFR_sobrecompradas': sobrecompradas,
        'IFR_sobrevendidas': sobrevendidas,
        'IFR_net': sobrecompradas - sobrevendidas,
        'IFR_media_geral': ifr_individual.mean(axis=1),
        'breadth_red': cat_red,
        'breadth_yellow': cat_yellow,
        'breadth_green': cat_green,
        'new_highs': new_highs,
        'new_lows': new_lows,
        'net_highs_lows': net_highs_lows,
        'mcclellan': mcclellan_osc,
        'macd_breadth': percentual_macd_bullish,
        'summation_index': summation_index,
        'cumulative_net_highs': cumulative_net_highs_lows
    })
    
    return df_amplitude

def analisar_retornos_por_faixa(df_analise, nome_coluna_indicador, passo, min_range, max_range, sufixo=''):
    bins = list(range(min_range, max_range + passo, passo))
    labels = [f'{i} a {i+passo}{sufixo}' for i in range(min_range, max_range, passo)]
    df_analise[f'faixa'] = pd.cut(df_analise[nome_coluna_indicador], bins=bins, labels=labels, right=False, include_lowest=True)
    colunas_retorno = [col for col in df_analise.columns if 'retorno_' in col]
    # observed=True is default in recent pandas but good to specify if using Categoricals
    grouped = df_analise.groupby(f'faixa', observed=False)
    media_resultados = grouped[colunas_retorno].mean()
    positivos = grouped[colunas_retorno].agg(lambda x: (x > 0).sum())
    totais = grouped[colunas_retorno].count()
    acerto_resultados = (positivos / totais * 100).fillna(0)
    return pd.concat([media_resultados, acerto_resultados], axis=1, keys=['Retorno Médio', 'Taxa de Acerto'])
