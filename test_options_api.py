"""
Teste de API para buscar dados de opções da B3
Este arquivo é para testar a obtenção de prêmios de opções em tempo real
antes de integrar no App.py principal
"""

import requests
import pandas as pd
from datetime import datetime, date
import yfinance as yf

# =============================================================================
# MÉTODO 1: Yahoo Finance (opcionalidades limitadas para B3)
# =============================================================================

def test_yfinance_options(ticker="BOVA11"):
    """
    Tenta buscar opções via yfinance.
    Nota: yfinance tem suporte limitado para opções brasileiras.
    """
    print(f"\n{'='*60}")
    print(f"Testando yfinance para {ticker}")
    print('='*60)
    
    try:
        full_ticker = f"{ticker}.SA" if not ticker.endswith(".SA") else ticker
        stock = yf.Ticker(full_ticker)
        
        # Tenta pegar opções
        if hasattr(stock, 'options') and stock.options:
            print(f"Vencimentos disponíveis: {stock.options}")
            
            # Pega o primeiro vencimento
            opt = stock.option_chain(stock.options[0])
            print(f"\nPUTs disponíveis:")
            print(opt.puts.head(10))
            return True
        else:
            print("❌ Yahoo Finance não retornou dados de opções para este ativo")
            return False
            
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False


# =============================================================================
# MÉTODO 2: B3 API (Opções listadas)
# =============================================================================

def test_b3_options_api():
    """
    Tenta buscar opções diretamente da API da B3.
    """
    print(f"\n{'='*60}")
    print("Testando API B3 para opções")
    print('='*60)
    
    # Endpoint da B3 para opções (pode variar)
    # A B3 não tem API pública fácil para opções em tempo real
    # Vamos tentar alguns endpoints conhecidos
    
    try:
        # Endpoint do InfoMoney (scraping básico)
        url = "https://www.infomoney.com.br/ferramentas/opcoes-de-acoes/"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            print("✅ Conseguiu acessar InfoMoney (requer parsing HTML)")
            print(f"   Status: {response.status_code}")
            print(f"   Tamanho: {len(response.content)} bytes")
            # Precisaria de BeautifulSoup para parsear
            return True
        else:
            print(f"❌ Status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False


# =============================================================================
# MÉTODO 3: StatusInvest API
# =============================================================================

def test_statusinvest_options(ticker="BOVA11"):
    """
    Tenta buscar opções via StatusInvest API.
    """
    print(f"\n{'='*60}")
    print(f"Testando StatusInvest para {ticker}")
    print('='*60)
    
    try:
        # API do StatusInvest para opções
        url = f"https://statusinvest.com.br/opcoes/{ticker.lower()}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            print("✅ Conseguiu acessar StatusInvest (requer parsing HTML)")
            print(f"   Status: {response.status_code}")
            print(f"   Tamanho: {len(response.content)} bytes")
            return True
        else:
            print(f"❌ Status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False


# =============================================================================
# MÉTODO 4: Fundamentus
# =============================================================================

def test_fundamentus_options():
    """
    Tenta buscar opções via Fundamentus.
    """
    print(f"\n{'='*60}")
    print("Testando Fundamentus para opções")
    print('='*60)
    
    try:
        url = "https://www.fundamentus.com.br/opcoes.php"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            print("✅ Conseguiu acessar Fundamentus")
            print(f"   Status: {response.status_code}")
            print(f"   Tamanho: {len(response.content)} bytes")
            
            # Tenta parsear tabela HTML
            try:
                tables = pd.read_html(response.content, decimal=',', thousands='.')
                if tables:
                    print(f"   Encontradas {len(tables)} tabelas")
                    for i, t in enumerate(tables):
                        print(f"\n   Tabela {i}: {t.shape}")
                        if len(t) > 0:
                            print(t.head(5))
                return True
            except Exception as parse_error:
                print(f"   ⚠️ Erro ao parsear: {parse_error}")
                return False
        else:
            print(f"❌ Status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False


# =============================================================================
# MÉTODO 5: B3 Cotações (Arquivo Diário)
# =============================================================================

def test_b3_daily_file():
    """
    A B3 disponibiliza arquivos diários com cotações.
    Este método tenta acessar o arquivo de opções.
    """
    print(f"\n{'='*60}")
    print("Testando arquivo diário B3")
    print('='*60)
    
    try:
        # A B3 disponibiliza arquivos BVBG086 para opções
        # URL típica (precisa verificar formato atual)
        today = date.today()
        date_str = today.strftime("%y%m%d")
        
        # Tenta várias URLs conhecidas
        urls = [
            f"https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_D{date_str}.ZIP",
            "https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/cotacoes/",
        ]
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        for url in urls:
            print(f"Tentando: {url[:60]}...")
            try:
                response = requests.head(url, headers=headers, timeout=5)
                print(f"   Status: {response.status_code}")
            except:
                print("   Timeout ou erro")
                
        return False  # Marcamos como false pois requer mais trabalho
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False


# =============================================================================
# MÉTODO 6: Opcoes.net.br (Sugerido pelo usuário)
# =============================================================================

def test_opcoes_net_br(ticker="BOVA11"):
    """
    Tenta buscar opções via opcoes.net.br
    URL: https://opcoes.net.br/opcoes/bovespa/BOVA11
    """
    print(f"\n{'='*60}")
    print(f"Testando opcoes.net.br para {ticker}")
    print('='*60)
    
    try:
        # URL da página de opções
        url = f"https://opcoes.net.br/opcoes/bovespa/{ticker.upper()}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7',
        }
        
        print(f"   URL: {url}")
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            print(f"✅ Conseguiu acessar opcoes.net.br")
            print(f"   Status: {response.status_code}")
            print(f"   Tamanho: {len(response.content)} bytes")
            
            # Tenta parsear tabelas HTML
            try:
                tables = pd.read_html(response.content, decimal=',', thousands='.')
                if tables:
                    print(f"\n   📊 Encontradas {len(tables)} tabelas!")
                    for i, t in enumerate(tables):
                        print(f"\n   === Tabela {i} ({t.shape[0]} linhas x {t.shape[1]} colunas) ===")
                        print(f"   Colunas: {list(t.columns)}")
                        if len(t) > 0:
                            print(t.head(10).to_string())
                    
                    # Verifica se encontrou dados de opções (PUT)
                    for i, t in enumerate(tables):
                        cols = [str(c).lower() for c in t.columns]
                        if any('strike' in c or 'prêmio' in c or 'premio' in c or 'exercício' in c for c in cols):
                            print(f"\n   🎯 Tabela {i} parece conter dados de opções!")
                            return True
                    
                    print("\n   ⚠️ Tabelas encontradas, mas estrutura não reconhecida como opções")
                    return True  # Ainda retorna True pois acessou
                else:
                    print("   ⚠️ Nenhuma tabela HTML encontrada")
                    return False
                    
            except Exception as parse_error:
                print(f"   ⚠️ Erro ao parsear tabelas: {parse_error}")
                
                # Tenta verificar se há JSON na página
                if 'application/json' in response.headers.get('Content-Type', ''):
                    print("   📄 Resposta parece ser JSON")
                    try:
                        import json
                        data = response.json()
                        print(f"   JSON keys: {list(data.keys()) if isinstance(data, dict) else 'lista'}")
                        return True
                    except:
                        pass
                
                return False
        else:
            print(f"❌ Status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False


# =============================================================================
# MAIN - Executar todos os testes
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTE DE APIs PARA DADOS DE OPÇÕES")
    print("="*60)
    
    results = {
        "yfinance": test_yfinance_options("BOVA11"),
        "b3_api": test_b3_options_api(),
        "statusinvest": test_statusinvest_options("BOVA11"),
        "fundamentus": test_fundamentus_options(),
        "b3_daily": test_b3_daily_file(),
        "opcoes_net_br": test_opcoes_net_br("BOVA11"),
    }
    
    print("\n" + "="*60)
    print("RESUMO DOS TESTES")
    print("="*60)
    
    for source, success in results.items():
        status = "✅ OK" if success else "❌ FALHOU"
        print(f"  {source:20s}: {status}")
    
    print("\n" + "="*60)
    print("PRÓXIMOS PASSOS")
    print("="*60)
    print("""
    Se algum método funcionou, podemos:
    1. Implementar parsing mais robusto
    2. Integrar no App.py
    3. Adicionar cache para não sobrecarregar APIs
    """)
