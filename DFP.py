import pandas as pd
import requests
import io
import zipfile
import os
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Torna ipywidgets opcional (para rodar fora do Jupyter)
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False

# --- CONSTANTES GLOBAIS ---
URL_BASE_CVM = 'https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC'
URL_CADASTRO = 'https://dados.cvm.gov.br/dados/CIA_ABERTA/CAD/DADOS/cad_cia_aberta.csv'
CACHE_DIR = Path(__file__).parent / '.cache'

# Colunas necessárias para otimizar leitura do CSV
COLUNAS_NECESSARIAS = ['CD_CVM', 'CD_CONTA', 'DS_CONTA', 'DT_FIM_EXERC', 'VL_CONTA']
DTYPE_MAP = {
    'CD_CVM': 'int32',
    'CD_CONTA': 'str',
    'DS_CONTA': 'str',
    'VL_CONTA': 'float64'
}

# --- FUNÇÃO OTIMIZADA PARA DOWNLOAD E PROCESSAMENTO ---

def baixar_e_processar_em_paralelo(anos: list, codigo_cvm: int, tipo_doc: str) -> dict:
    """
    Baixa e processa arquivos DFP ou ITR em paralelo para dados consolidados e individuais.
    OTIMIZADO: Usa listas para acumulação e concatena apenas no final.

    Args:
        anos (list): Lista de anos para processar.
        codigo_cvm (int): Código CVM da empresa.
        tipo_doc (str): 'DFP' ou 'ITR'.
    """
    print(f"\n>>> Iniciando busca otimizada de dados {tipo_doc} para os anos {min(anos)}-{max(anos)}...")

    # Estrutura para acumular DFs em listas (OTIMIZAÇÃO: evita concat em loop)
    acumulador = {
        'DRE': {'con': [], 'ind': []},
        'BPA': {'con': [], 'ind': []},
        'BPP': {'con': [], 'ind': []},
        'DFC_MI': {'con': [], 'ind': []}
    }

    session = requests.Session()

    def baixar_e_filtrar_ano(ano):
        url_zip = f"{URL_BASE_CVM}/{tipo_doc}/DADOS/{tipo_doc.lower()}_cia_aberta_{ano}.zip"
        dfs_ano = {chave: {'con': [], 'ind': []} for chave in acumulador.keys()}

        try:
            response = session.get(url_zip, stream=True, timeout=30)
            response.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                for nome_arquivo in z.namelist():
                    # Identifica se o arquivo é consolidado ou individual
                    tipo_relatorio = None
                    if '_con' in nome_arquivo:
                        tipo_relatorio = 'con'
                    elif '_ind' in nome_arquivo:
                        tipo_relatorio = 'ind'

                    if not tipo_relatorio:
                        continue  # Pula arquivos que não são 'con' nem 'ind'

                    chave = None
                    if 'DFC_MI' in nome_arquivo:
                        chave = 'DFC_MI'
                    elif 'DRE' in nome_arquivo:
                        chave = 'DRE'
                    elif 'BPA' in nome_arquivo:
                        chave = 'BPA'
                    elif 'BPP' in nome_arquivo:
                        chave = 'BPP'

                    if chave:
                        with z.open(nome_arquivo) as f:
                            # OTIMIZAÇÃO: usecols e dtype para leitura mais rápida
                            leitor_csv = pd.read_csv(
                                f,
                                sep=';',
                                encoding='ISO-8859-1',
                                usecols=lambda col: col in COLUNAS_NECESSARIAS,
                                dtype=DTYPE_MAP,
                                chunksize=100000
                            )
                            for chunk in leitor_csv:
                                df_filtrado = chunk[chunk['CD_CVM'] == codigo_cvm]
                                if not df_filtrado.empty:
                                    dfs_ano[chave][tipo_relatorio].append(df_filtrado)

            # Concatena todos os chunks filtrados para o ano
            for chave in dfs_ano:
                for tipo in ['con', 'ind']:
                    if dfs_ano[chave][tipo]:
                        dfs_ano[chave][tipo] = pd.concat(dfs_ano[chave][tipo], ignore_index=True)
                    else:
                        dfs_ano[chave][tipo] = pd.DataFrame()

            print(f"   [OK] Ano {ano} processado.")
            return dfs_ano

        except requests.exceptions.HTTPError:
            return None
        except Exception as e:
            print(f"   [ERRO] Erro ao processar ano {ano}: {e}")
            return None

    # OTIMIZAÇÃO: número dinâmico de workers
    num_workers = min(8, len(anos), os.cpu_count() or 4)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_ano = {executor.submit(baixar_e_filtrar_ano, ano): ano for ano in anos}

        for future in as_completed(future_to_ano):
            resultado_ano = future.result()
            if resultado_ano:
                for chave in acumulador.keys():
                    for tipo in ['con', 'ind']:
                        df_ano = resultado_ano[chave][tipo]
                        if not isinstance(df_ano, pd.DataFrame):
                            continue
                        if not df_ano.empty:
                            # OTIMIZAÇÃO: acumula em lista ao invés de concat imediato
                            acumulador[chave][tipo].append(df_ano)

    # OTIMIZAÇÃO: concatenação única no final
    demonstrativos = {
        'DRE': {'con': pd.DataFrame(), 'ind': pd.DataFrame()},
        'BPA': {'con': pd.DataFrame(), 'ind': pd.DataFrame()},
        'BPP': {'con': pd.DataFrame(), 'ind': pd.DataFrame()},
        'DFC_MI': {'con': pd.DataFrame(), 'ind': pd.DataFrame()}
    }
    
    for chave in acumulador.keys():
        for tipo in ['con', 'ind']:
            if acumulador[chave][tipo]:
                demonstrativos[chave][tipo] = pd.concat(acumulador[chave][tipo], ignore_index=True)

    print(f"<<< Busca de dados {tipo_doc} finalizada.")
    return demonstrativos


# --- FUNÇÃO DE FORMATAÇÃO UNIFICADA (OTIMIZAÇÃO) ---

def formatar_demonstrativo(df: pd.DataFrame, tipo: str, is_trimestral: bool = False) -> pd.DataFrame:
    """
    Formata demonstrativos financeiros em formato pivotado.
    OTIMIZADO: Função única para dados anuais e trimestrais.
    
    Args:
        df: DataFrame com os dados brutos
        tipo: Tipo do demonstrativo ('DRE', 'BP', 'DFC')
        is_trimestral: Se True, formata para períodos trimestrais (ex: 2024T1)
    """
    if df.empty:
        return pd.DataFrame()
    
    df_processed = df.drop_duplicates(subset=['DT_FIM_EXERC', 'CD_CONTA'], keep='last').copy()
    df_processed['DT_FIM_EXERC'] = pd.to_datetime(df_processed['DT_FIM_EXERC'])
    
    if is_trimestral:
        df_processed['PERIODO'] = (
            df_processed['DT_FIM_EXERC'].dt.strftime('%Y') + 'T' + 
            df_processed['DT_FIM_EXERC'].dt.quarter.astype(str)
        )
        coluna_tempo = 'PERIODO'
    else:
        df_processed['ANO'] = df_processed['DT_FIM_EXERC'].dt.year
        coluna_tempo = 'ANO'
    
    df_final = df_processed[['CD_CONTA', 'DS_CONTA', coluna_tempo, 'VL_CONTA']]
    df_pivot = df_final.pivot_table(
        index=['CD_CONTA', 'DS_CONTA'],
        columns=coluna_tempo,
        values='VL_CONTA',
        aggfunc='first'
    ).reset_index()
    
    df_pivot = df_pivot.sort_values(by='CD_CONTA')
    df_pivot.columns = [str(col) for col in df_pivot.columns]
    return df_pivot


def consolidar_contas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consolida linhas com mesmo CD_CONTA mas descrições diferentes.
    Mantém a descrição mais recente (com dados mais recentes) e soma os valores.
    """
    if df.empty or 'CD_CONTA' not in df.columns:
        return df
    
    # Identifica colunas de valores (anos/períodos)
    colunas_fixas = ['CD_CONTA', 'DS_CONTA']
    colunas_valores = [c for c in df.columns if c not in colunas_fixas]
    
    if not colunas_valores:
        return df
    
    # Agrupa por CD_CONTA e pega a primeira descrição (mais comum)
    # Soma os valores (ou pega o primeiro não-nulo)
    df_grouped = df.groupby('CD_CONTA', as_index=False).agg(
        {
            'DS_CONTA': 'first',  # Mantém primeira descrição
            **{col: lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else None 
               for col in colunas_valores}
        }
    )
    
    # Reordena colunas
    df_grouped = df_grouped[['CD_CONTA', 'DS_CONTA'] + colunas_valores]
    return df_grouped.sort_values(by='CD_CONTA').reset_index(drop=True)


def calcular_ltm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula colunas LTM (Last Twelve Months) para DRE e DFC.
    
    Fórmula para LTM do trimestre Q em ano N:
    LTM(Qn) = Ano(N-1) - soma(Q1..Qn de N-1) + soma(Q1..Qn de N)
    
    Para Q4, LTM = Ano (já que Q4 fecha o ano)
    """
    if df.empty:
        return df
    
    df = df.copy()
    colunas = list(df.columns)
    
    # Identifica colunas de anos (ex: '2024') e trimestres (ex: '2024T3')
    anos = sorted([c for c in colunas if c.isdigit() and len(c) == 4])
    trimestres = sorted([c for c in colunas if 'T' in c and len(c) == 6])
    
    if not anos or not trimestres:
        return df  # Sem dados suficientes para LTM
    
    # Extrai anos únicos dos trimestres
    anos_trim = sorted(set([t[:4] for t in trimestres]))
    
    ltm_cols = {}
    
    for ano in anos_trim:
        ano_int = int(ano)
        ano_anterior = str(ano_int - 1)
        
        # Verifica se temos o ano anterior
        if ano_anterior not in anos:
            continue
        
        for q in [1, 2, 3, 4]:
            col_trim = f"{ano}T{q}"
            
            if col_trim not in trimestres:
                continue
            
            if q == 4:
                # Para Q4, LTM é o próprio ano (quando disponível)
                if ano in anos:
                    ltm_cols[f"LTM_{ano}T4"] = df[ano].copy()
            else:
                # Precisa calcular: Ano(N-1) - soma(Q1..Qn de N-1) + soma(Q1..Qn de N)
                # Trimestres do ano anterior a subtrair
                trims_ant_sub = [f"{ano_anterior}T{i}" for i in range(1, q + 1)]
                # Trimestres do ano atual a somar
                trims_atual_add = [f"{ano}T{i}" for i in range(1, q + 1)]
                
                # Verifica disponibilidade
                trims_ant_disponiveis = [t for t in trims_ant_sub if t in trimestres]
                trims_atual_disponiveis = [t for t in trims_atual_add if t in trimestres]
                
                if len(trims_ant_disponiveis) == q and len(trims_atual_disponiveis) == q:
                    ltm = df[ano_anterior].copy()
                    
                    for t in trims_ant_disponiveis:
                        ltm = ltm - df[t].fillna(0)
                    
                    for t in trims_atual_disponiveis:
                        ltm = ltm + df[t].fillna(0)
                    
                    ltm_cols[f"LTM_{col_trim}"] = ltm
    
    # Adiciona colunas LTM ao DataFrame
    for col_name, col_data in ltm_cols.items():
        df[col_name] = col_data
    
    # Reordena: CD_CONTA, DS_CONTA, Anos, Trimestres, LTMs
    colunas_fixas = ['CD_CONTA', 'DS_CONTA']
    ltm_colunas = sorted([c for c in df.columns if c.startswith('LTM_')])
    
    nova_ordem = colunas_fixas + anos + sorted(trimestres) + ltm_colunas
    nova_ordem = [c for c in nova_ordem if c in df.columns]
    
    return df[nova_ordem]


def combinar_dados(df_dfp: pd.DataFrame, df_itr: pd.DataFrame, aplicar_ltm: bool = False) -> pd.DataFrame:
    """
    Helper para combinar dados anuais (DFP) e trimestrais (ITR).
    Aplica consolidação de contas duplicadas e opcionalmente LTM.
    """
    if not df_dfp.empty and not df_itr.empty:
        df_merged = pd.merge(df_dfp, df_itr, on=['CD_CONTA', 'DS_CONTA'], how='outer')
    elif not df_dfp.empty:
        df_merged = df_dfp
    elif not df_itr.empty:
        df_merged = df_itr
    else:
        return pd.DataFrame()
    
    # Consolida contas com mesmo CD_CONTA
    df_consolidado = consolidar_contas(df_merged)
    
    # Aplica LTM se solicitado (para DRE e DFC)
    if aplicar_ltm:
        df_consolidado = calcular_ltm(df_consolidado)
    
    return df_consolidado


# --- FUNÇÃO PRINCIPAL ---

def main_analysis(nome_empresa: str, codigo_cvm: int):
    """
    Função principal que orquestra a execução do script.
    """
    print(f"--- INICIANDO ANÁLISE PARA: {nome_empresa} (CVM: {codigo_cvm}) ---")
    ano_atual = datetime.now().year

    anos_dfp = list(range(ano_atual - 20, ano_atual + 1))
    dados_cia_dfp = baixar_e_processar_em_paralelo(anos_dfp, codigo_cvm, 'DFP')

    anos_itr = list(range(ano_atual - 5, ano_atual + 1))
    demonstrativos_itr_raw = baixar_e_processar_em_paralelo(anos_itr, codigo_cvm, 'ITR')

    print("\n3. Formatando e organizando os dados...")

    dataframes_to_export = {}

    for tipo in ['con', 'ind']:
        # Processamento DFP (usando função unificada)
        bp_dfp_raw = pd.concat([dados_cia_dfp['BPA'][tipo], dados_cia_dfp['BPP'][tipo]], ignore_index=True)
        dre_formatado_dfp = formatar_demonstrativo(dados_cia_dfp['DRE'][tipo], 'DRE', is_trimestral=False)
        bp_formatado_dfp = formatar_demonstrativo(bp_dfp_raw, 'BP', is_trimestral=False)
        dfc_formatado_dfp = formatar_demonstrativo(dados_cia_dfp['DFC_MI'][tipo], 'DFC', is_trimestral=False)

        # Processamento ITR (usando função unificada)
        bp_itr_raw = pd.concat([demonstrativos_itr_raw['BPA'][tipo], demonstrativos_itr_raw['BPP'][tipo]], ignore_index=True)
        dre_formatado_itr = formatar_demonstrativo(demonstrativos_itr_raw['DRE'][tipo], 'DRE', is_trimestral=True)
        bp_formatado_itr = formatar_demonstrativo(bp_itr_raw, 'BP', is_trimestral=True)
        dfc_formatado_itr = formatar_demonstrativo(demonstrativos_itr_raw['DFC_MI'][tipo], 'DFC', is_trimestral=True)

        # Combinação e armazenamento para exportação
        # LTM aplicado apenas a DRE e DFC (não faz sentido para BP)
        dataframes_to_export[f'DRE_{tipo}'] = combinar_dados(dre_formatado_dfp, dre_formatado_itr, aplicar_ltm=True)
        dataframes_to_export[f'BP_{tipo}'] = combinar_dados(bp_formatado_dfp, bp_formatado_itr, aplicar_ltm=False)
        dataframes_to_export[f'DFC_{tipo}'] = combinar_dados(dfc_formatado_dfp, dfc_formatado_itr, aplicar_ltm=True)

    print("\n4. Exportando dados para o arquivo Excel...")
    nome_arquivo_excel = f"Analise_Financeira_{nome_empresa.replace(' ', '_').replace('/', '')}.xlsx"

    if not any(not df.empty for df in dataframes_to_export.values()):
        print("\nErro: Nenhum dado (DFP ou ITR) disponível para exportar para Excel.")
        return

    with pd.ExcelWriter(nome_arquivo_excel, engine='openpyxl') as writer:
        for sheet_name, df_to_export in dataframes_to_export.items():
            if not df_to_export.empty:
                df_to_export.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"   Exportado: {sheet_name}")

        info_data = {
            'Informacao': ['Empresa Analisada', 'Codigo CVM', 'Fonte de Dados', 'Data da Extracao'],
            'Valor': [nome_empresa, codigo_cvm, URL_BASE_CVM, datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        }
        pd.DataFrame(info_data).to_excel(writer, sheet_name='Info', index=False)

    print(f"\n[SUCESSO] Os dados foram salvos em '{os.path.abspath(nome_arquivo_excel)}'")


# --- CACHE PARA CADASTRO ---

def carregar_cadastro_com_cache(max_age_hours: int = 24) -> pd.DataFrame:
    """
    Carrega cadastro de empresas com cache local.
    OTIMIZAÇÃO: Evita downloads repetidos.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / 'cad_cia_aberta.parquet'
    
    # Verifica se cache existe e é recente
    if cache_file.exists():
        cache_age = datetime.now().timestamp() - cache_file.stat().st_mtime
        if cache_age < max_age_hours * 3600:
            print("[CACHE] Usando cadastro em cache...")
            return pd.read_parquet(cache_file)
    
    # Baixa e salva em cache
    print(">>> Baixando cadastro de empresas...")
    df = pd.read_csv(URL_CADASTRO, sep=';', encoding='ISO-8859-1')
    df = df[df['SIT'] == 'ATIVO']
    df['DT_REG'] = pd.to_datetime(df['DT_REG'])
    df = df.sort_values(by='DT_REG', ascending=False)
    df = df.drop_duplicates(subset=['DENOM_SOCIAL'], keep='first')
    
    df.to_parquet(cache_file, index=False)
    print(f"[CACHE] Cadastro salvo em cache ({len(df)} empresas)")
    return df


# --- FUNÇÕES DE INTERFACE (Jupyter only) ---

if JUPYTER_AVAILABLE:
    @widgets.interact_manual
    def carregar_dados_cadastrais():
        global cad_df_global
        try:
            cad_df_global = carregar_cadastro_com_cache()
            print(f"✅ Lista de empresas carregada! {len(cad_df_global)} empresas únicas ativas encontradas.")
            iniciar_busca_interativa()
        except Exception as e:
            print(f"Erro ao carregar o cadastro de empresas: {e}")
            cad_df_global = pd.DataFrame()


    # --- LÓGICA INTERATIVA ---
    search_box = widgets.Text(
        value='',
        placeholder='Digite o nome da empresa (mín. 3 caracteres)',
        description='Buscar:',
        disabled=False,
        layout=widgets.Layout(width='50%')
    )
    output_area = widgets.Output()
    main_container = widgets.VBox()


    def on_company_select(b):
        main_container.children = []
        clear_output(wait=True)
        nome = b.description
        codigo_cvm = b.codigo_cvm
        main_analysis(nome_empresa=nome, codigo_cvm=codigo_cvm)


    def on_text_change(change):
        with output_area:
            clear_output(wait=True)
            query = change.new.strip().upper()
            if len(query) < 3:
                print("Digite pelo menos 3 caracteres para buscar.")
                return
            if 'cad_df_global' in globals() and not cad_df_global.empty:
                resultados = cad_df_global[cad_df_global['DENOM_SOCIAL'].str.contains(query, case=False, na=False)].head(15)
                if not resultados.empty:
                    print(f"Resultados encontrados para '{query}': (clique para analisar)")
                    botoes = []
                    for index, row in resultados.iterrows():
                        btn = widgets.Button(description=row['DENOM_SOCIAL'], layout=widgets.Layout(width='90%'))
                        btn.codigo_cvm = row['CD_CVM']
                        btn.on_click(on_company_select)
                        botoes.append(btn)
                    display(widgets.VBox(botoes))
                else:
                    print(f"Nenhuma empresa encontrada para '{query}'.")


    def iniciar_busca_interativa():
        clear_output()
        search_box.observe(on_text_change, names='value')
        titulo = widgets.HTML("<h3>Selecione a Empresa para Análise Financeira</h3>")
        main_container.children = [titulo, search_box, output_area]
        display(main_container)


# --- TESTE DIRETO (sem interface Jupyter) ---

def teste_direto():
    """
    Função de teste que executa análise para uma empresa conhecida.
    Útil para testar fora do ambiente Jupyter.
    """
    print("=" * 60)
    print("TESTE DIRETO - Analise Financeira CVM")
    print("=" * 60)
    
    # Petrobras - código CVM conhecido
    nome_empresa = "PETROBRAS"
    codigo_cvm = 9512
    
    main_analysis(nome_empresa=nome_empresa, codigo_cvm=codigo_cvm)


if __name__ == "__main__":
    teste_direto()
elif JUPYTER_AVAILABLE:
    # --- PONTO DE PARTIDA (Jupyter) ---
    print("Execute a célula e clique no botão abaixo para começar.")
    carregar_dados_cadastrais