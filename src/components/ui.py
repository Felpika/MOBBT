
import streamlit as st
import pandas as pd

def display_metrics_row(metrics):
    """
    Exibe uma linha de métricas.
    metrics: lista de dicts com keys 'label', 'value', 'delta', 'help'
    """
    cols = st.columns(len(metrics))
    for col, metric in zip(cols, metrics):
        col.metric(
            label=metric.get('label'),
            value=metric.get('value'),
            delta=metric.get('delta'),
            help=metric.get('help')
        )

def format_currency(value):
    return f"R$ {value:,.2f}"

def format_percentage(value):
    return f"{value:.2f}%"
