
import plotly.io as pio

# --- THEME CONSTANTS (BROKEBERG) ---
COLORS = {
    'VERDE_NEON': '#39E58C',
    'AMARELO_OURO': '#FFB302',
    'CIANO_NEON': '#00D4FF',
    'VERMELHO_NEON': '#FF4B4B',
    'FUNDO_ESCURO': '#050505',
    'FUNDO_CARDS': '#161B22',
    'TEXTO_PRINCIPAL': '#F0F6FC',
    'TEXTO_SECUNDARIO': '#C9D1D9',
    'GRADE_SUTIL': '#30363D',
}

def configurar_tema_brokeberg():
    """Configures the custom Plotly theme 'brokeberg'."""
    brokeberg_template = pio.templates["plotly_dark"]  # Baseia-se no dark para facilitar
    
    # Customiza o Layout Global
    brokeberg_template.layout.update(
        paper_bgcolor=COLORS['FUNDO_ESCURO'],
        plot_bgcolor=COLORS['FUNDO_ESCURO'],
        font={'color': COLORS['TEXTO_SECUNDARIO'], 'family': "Segoe UI, sans-serif"},
        title={'font': {'color': COLORS['TEXTO_PRINCIPAL'], 'size': 20}},
        
        # Eixo X
        xaxis={
            'gridcolor': COLORS['GRADE_SUTIL'],
            'linecolor': COLORS['GRADE_SUTIL'],
            'zerolinecolor': COLORS['GRADE_SUTIL'],
            'tickfont': {'color': COLORS['TEXTO_SECUNDARIO']}
        },
        # Eixo Y
        yaxis={
            'gridcolor': COLORS['GRADE_SUTIL'],
            'linecolor': COLORS['GRADE_SUTIL'],
            'zerolinecolor': COLORS['GRADE_SUTIL'],
            'tickfont': {'color': COLORS['TEXTO_SECUNDARIO']}
        },
        # Cores padrão para linhas (Ciclo de cores)
        colorway=[
            COLORS['CIANO_NEON'], 
            COLORS['VERDE_NEON'], 
            COLORS['AMARELO_OURO'], 
            COLORS['VERMELHO_NEON'], 
            '#AB47BC', 
            '#5C6BC0'
        ]
    )

    # Registra e define como padrão
    pio.templates["brokeberg"] = brokeberg_template
    pio.templates.default = "brokeberg"
