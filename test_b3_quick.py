# -*- coding: utf-8 -*-
"""
Teste da API B3 rapinegocios para opcoes
"""
import requests
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Teste o link que o usuario encontrou
url = "https://arquivos.b3.com.br/rapinegocios/tickercsv/BOVAN159/2026-01-05"

print(f"Testando URL: {url}")

try:
    r = requests.get(url, timeout=15)
    print(f"Status: {r.status_code}")
    print(f"Content-Type: {r.headers.get('Content-Type', 'N/A')}")
    print(f"Tamanho: {len(r.content)} bytes")
    
    if r.status_code == 200:
        print("\n=== CONTEUDO ===")
        print(r.text[:2000])
    else:
        print(f"Erro: {r.text[:500]}")
        
except Exception as e:
    print(f"Erro: {e}")
