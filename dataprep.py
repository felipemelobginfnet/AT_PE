import requests
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import google.generativeai as genai
from datetime import datetime
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

URL_BASE = "https://dadosabertos.camara.leg.br/api/v2/"
GEMINI_TOKEN = os.getenv("GEMINI_TOKEN")

os.makedirs("data", exist_ok=True)
os.makedirs("docs", exist_ok=True)

def configurar_modelo():
    """Configura o modelo de linguagem."""
    try:
        genai.configure(api_key=GEMINI_TOKEN)
        config_seguranca = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
        }
        
        config_geracao = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40
        }
        
        modelo = genai.GenerativeModel("gemini-pro", 
                                     safety_settings=config_seguranca,
                                     generation_config=config_geracao)
        return modelo
    except Exception as e:
        logger.error(f"Erro ao configurar modelo: {e}")
        return None

def gerar_resposta_modelo(modelo, prompt):
    """Gera resposta do modelo de linguagem."""
    if not modelo:
        logger.error("Modelo não configurado")
        return None
    
    try:
        resposta = modelo.generate_content(prompt)
        if hasattr(resposta, 'text'):
            return resposta.text
        return None
    except Exception as e:
        logger.error(f"Erro ao gerar resposta: {e}")
        return None

def coletar_deputados():
    """Coleta dados dos deputados atuais."""
    try:
        url_deputados = f"{URL_BASE}deputados"
        resposta = requests.get(url_deputados)
        resposta.raise_for_status()
        deputados = pd.json_normalize(resposta.json()["dados"])
        deputados.to_parquet("data/deputados.parquet", index=False)
        return deputados
    except Exception as e:
        logger.error(f"Erro ao coletar dados dos deputados: {e}")
        return pd.DataFrame()

def gerar_grafico_pizza(deputados):
    """Gera gráfico de pizza para distribuição partidária."""
    try:
        plt.figure(figsize=(12, 8))
        distribuicao = deputados['siglaPartido'].value_counts()
        plt.pie(distribuicao, labels=distribuicao.index, autopct='%1.1f%%')
        plt.title('Distribuição de Deputados por Partido')
        plt.savefig('docs/distribuicao_deputados.png')
        plt.close()
    except Exception as e:
        logger.error(f"Erro ao gerar gráfico de pizza: {e}")

def prompt_grafico_pizza():
    """Prompt para geração do código do gráfico de pizza."""
    return """
    Como um especialista em visualização de dados com Python, crie um código que:
    
    1. Leia o arquivo 'data/deputados.parquet'
    2. Calcule o total de deputados por partido
    3. Crie um gráfico de pizza mostrando:
       - Distribuição percentual por partido
       - Legendas claras
       - Cores distintas
       - Valores percentuais no gráfico
    4. Salve o gráfico como 'docs/distribuicao_deputados.png'
    
    O código deve usar matplotlib e ser completo, incluindo todas as importações necessárias.
    """

def prompt_analise_distribuicao(dados_partidos):
    """Prompt para análise da distribuição partidária."""
    prompt = {
        "dados": dados_partidos.to_dict(),
        "persona": "Analista Político Sênior especializado em governabilidade e comportamento legislativo",
        "exemplo": "A fragmentação partidária pode dificultar a formação de maiorias estáveis, exigindo maior articulação política",
        "instrucoes": """
        Analise a distribuição de deputados por partido e explique:
        1. Como essa distribuição afeta a governabilidade?
        2. Quais são as implicações para formação de coalizões?
        3. Como isso impacta o processo legislativo?
        
        Forneça insights estratégicos e práticos.
        """
    }
    return json.dumps(prompt, ensure_ascii=False)

def coletar_despesas(deputados):
    """Coleta dados de despesas."""
    try:
        despesas = []
        for _, deputado in deputados.iterrows():
            url = f"{URL_BASE}deputados/{deputado['id']}/despesas"
            resposta = requests.get(url)
            if resposta.ok:
                dados = resposta.json()["dados"]
                for despesa in dados:
                    despesa["idDeputado"] = deputado["id"]
                despesas.extend(dados)
        
        if not despesas:
            return pd.DataFrame()
            
        df_despesas = pd.DataFrame(despesas)
        df_despesas['dataDocumento'] = pd.to_datetime(df_despesas['dataDocumento']).dt.date
        despesas_agrupadas = df_despesas.groupby(
            ["dataDocumento", "idDeputado", "tipoDespesa"]
        ).sum(numeric_only=True).reset_index()
        
        despesas_agrupadas.to_parquet("data/serie_despesas_diarias_deputados.parquet", index=False)
        return despesas_agrupadas
    except Exception as e:
        logger.error(f"Erro ao coletar despesas: {e}")
        return pd.DataFrame()

def analisar_despesas(despesas):
    """Analisa dados de despesas."""
    try:
        if despesas.empty:
            return None
            
        total_por_deputado = despesas.groupby('idDeputado')['valorDocumento'].sum().sort_values(ascending=False)
        media_diaria = despesas.groupby('dataDocumento')['valorDocumento'].mean()
        total_por_tipo = despesas.groupby('tipoDespesa')['valorDocumento'].sum().sort_values(ascending=False)
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        total_por_deputado.head(10).plot(kind='bar')
        plt.title('Top 10 Deputados por Gastos')
        plt.xticks(rotation=45)
        
        plt.subplot(132)
        plt.plot(media_diaria.index, media_diaria.values)
        plt.title('Média de Gastos Diários')
        plt.xticks(rotation=45)
        
        plt.subplot(133)
        total_por_tipo.head(5).plot(kind='pie', autopct='%1.1f%%')
        plt.title('Distribuição por Tipo de Despesa')
        
        plt.tight_layout()
        plt.savefig('docs/analise_despesas.png')
        plt.close()
        
        return {
            'total_por_deputado': total_por_deputado.to_dict(),
            'media_diaria': {str(k): v for k, v in media_diaria.to_dict().items()},
            'total_por_tipo': total_por_tipo.to_dict()
        }
    except Exception as e:
        logger.error(f"Erro ao analisar despesas: {e}")
        return None

def prompt_analise_despesas():
    """Prompt para análise de despesas."""
    return """
    Como um analista de dados especializado em gastos públicos, crie um código Python que:

    1. Leia o arquivo 'data/serie_despesas_diarias_deputados.parquet'
    2. Realize as seguintes análises:
       a. Ranking dos deputados por total de despesas
       b. Análise temporal das despesas médias diárias
       c. Identificação de padrões de gastos por tipo de despesa
    
    O código deve incluir:
    - Visualizações adequadas para cada análise
    - Cálculos estatísticos relevantes
    - Comentários explicativos
    - Salvamento dos resultados em formato adequado
    """

def prompt_insights_despesas(resultados_analise):
    """Prompt para geração de insights sobre despesas."""
    prompt = {
        "conhecimento_previo": resultados_analise,
        "instrucoes": """
        Como analista especializado em gastos públicos, baseado nos dados analisados:
        1. Quais são os principais padrões de gastos identificados?
        2. Existem anomalias ou pontos de atenção?
        3. Que recomendações podem ser feitas para otimização dos gastos?
        
        Forneça insights actionable e baseados em evidências.
        """
    }
    return json.dumps(prompt, ensure_ascii=False)

def coletar_proposicoes(data_inicio=datetime(2024, 8, 1), data_fim=datetime(2024, 8, 30)):
    """Coleta proposições por tema."""
    try:
        temas = {
            "40": "Economia",
            "46": "Educação",
            "62": "Ciência, Tecnologia e Inovação"
        }
        
        proposicoes = []
        for codigo, tema in temas.items():
            url = f"{URL_BASE}proposicoes?dataInicio={data_inicio.strftime('%Y-%m-%d')}&dataFim={data_fim.strftime('%Y-%m-%d')}&codTema={codigo}"
            resposta = requests.get(url)
            if resposta.ok:
                dados = resposta.json()["dados"][:10]
                for prop in dados:
                    prop["tema"] = tema
                proposicoes.extend(dados)
        
        df_proposicoes = pd.DataFrame(proposicoes)
        df_proposicoes.to_parquet("data/proposicoes_deputados.parquet", index=False)
        return df_proposicoes
    except Exception as e:
        logger.error(f"Erro ao coletar proposições: {e}")
        return pd.DataFrame()

def prompt_sumarizacao_proposicoes(texto_proposicao):
    """Prompt para sumarização de proposições."""
    return f"""
    Como especialista em análise legislativa, crie um resumo conciso e informativo da seguinte proposição:

    {texto_proposicao}

    O resumo deve:
    1. Identificar o objetivo principal
    2. Destacar os pontos-chave
    3. Manter-se objetivo e claro
    4. Ter no máximo 3 parágrafos
    
    Foque nos aspectos mais relevantes para o entendimento da proposta.
    """

def criar_base_vetorial(proposicoes):
    """Cria base vetorial para o assistente."""
    try:
        if proposicoes.empty:
            return
            
        modelo = SentenceTransformer('neuralmind/bert-base-portuguese-cased')
        embeddings = modelo.encode(proposicoes["ementa"].tolist())
        
        dimensao = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimensao)
        index.add(embeddings.astype('float32'))
        
        proposicoes['embedding'] = embeddings.tolist()
        proposicoes.to_parquet("data/proposicoes_vetorizadas.parquet", index=False)
        faiss.write_index(index, "data/proposicoes_index.faiss")
    except Exception as e:
        logger.error(f"Erro ao criar base vetorial: {e}")




def criar_prompt_visao_geral():
    return """
"Como especialista em desenvolvimento Python e Streamlit, crie um dashboard.py seguindo estas etapas:

Etapa 1 - Análise Inicial:
- Qual é a estrutura básica necessária para o dashboard?
- Quais bibliotecas precisamos importar?
- Como organizaremos as diferentes abas?

Etapa 2 - Desenvolvimento da Aba Visão Geral:
- Como implementar um título e descrição informativos?
- Como carregar e exibir o texto do arquivo config.yaml?
- Como criar o gráfico de barras dos dados em docs/distribuicao_deputados.png?
- Como mostrar os insights do arquivo data/insights_distribuicao_deputados.json?

Etapa 3 - Implementação Final:
- Como estruturar o código final com todas as funcionalidades?
- Como garantir que o código seja bem comentado e organizado?
- Como implementar tratamento de erros adequado?

Forneça o código completo em Python que implementa estas funcionalidades.
Lembre que é necessário que este código funcione 100%, então o código criado não deve precisar de intervenção humana para rodar. Todos os comentários
devem estar devidamente dentro da sintaxe do python para não interferir ou impossibilitar no seu funcionamento,
deve estar com tratamento de erros apropriado e boas práticas do Streamlit.
Além de revisar para saber se tudo que foi pedido foi criado."
"""

def criar_prompt_abas_adicionais():
    """
    Cria o prompt usando técnica Batch Prompting para as abas Despesas e Proposições
    """
    return """
"Como especialista em desenvolvimento Python e Streamlit, adicione ao dashboard.py as seguintes funcionalidades em lote:

Aba Despesas:
1. Carregar e mostrar insights do arquivo data/insights_despesas_deputados.json
2. Criar caixa de seleção para escolha do deputado
3. Gerar gráfico de barras temporal usando data/serie_despesas_diarias_deputados.parquet
4. Garantir formatação e estilo adequados

Aba Proposições:
1. Criar tabela com dados de data/proposicoes_deputados.parquet
2. Exibir resumo do arquivo data/sumarizacao_proposicoes.json
3. Utilizar os arquivos data\insights_despesas_deputados.json, data\insights_distribuicao_deputados.json, 
data\proposicoes_deputados.parquet, data\proposicoes_index.faiss, data\proposicoes_vetorizadas.parquet, data\serie_despesas_diarias_deputados.parquet, 
data\sumarizacao_proposicoes.json como contexto e fonte de informação para a resposta de um modelo de LLM
4. Adicionar filtros interativos
5. Manter organização visual consistente
6. Utilizar exatamente esta configuração para um modelo de LLM para responder perguntas feitas pelo usuário final
genai.configure(api_key=GEMINI_TOKEN)
        configuracoes_seguranca = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
        }
        
        configuracoes_geracao = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40
        }
        
        modelo = genai.GenerativeModel(
            "gemini-pro",
            safety_settings=configuracoes_seguranca,
            generation_config=configuracoes_geracao
        )
        return modelo
    except Exception as erro:
        print(f"Erro ao configurar modelo: {erro}")
        return None
 7. Utilizar técnicas de self ask antes de responder.      

Requisitos:
- Código bem comentado e organizado
- Tratamento adequado de erros
- Seguir boas práticas do Streamlit
- Manter padrão visual entre as abas

Forneça o código completo em Python que implementa estas funcionalidades de uma vez.
Lembre que é necessário que este código funcione 100%, então o código criado não deve precisar de intervenção humana para rodar. Todos os comentários
devem estar devidamente dentro da sintaxe do python para não interferir ou impossibilitar no seu funcionamento,
deve estar com tratamento de erros apropriado e boas práticas do Streamlit.
Além de revisar para saber se tudo que foi pedido foi criado."

"""

def criar_prompt_assistente():
    """
    Cria o prompt usando técnica Self-Ask para o assistente virtual
    """
    return """
"Você é um assistente especialista em dados da Câmara dos Deputados. Para cada pergunta, siga este processo:

1. Análise Inicial:
   Pergunto a mim mesmo:
   - Qual o tipo específico de informação solicitada?
   - Quais dados da base vetorial FAISS são relevantes?
   - Que análises preciso realizar?

2. Verificação:
   Pergunto a mim mesmo:
   - Como confirmar a precisão desta informação?
   - Existem exceções a considerar?
   - Que contexto adicional é importante?

3. Apresentação:
   Pergunto a mim mesmo:
   - Como apresentar os dados claramente?
   - Quais detalhes complementares são relevantes?
   - Há padrões importantes a destacar?

Use o modelo neuralmind/bert-base-portuguese-cased e a base FAISS para processar as informações.

Gere o código Python que implementa este assistente virtual.
Lembre que é necessário que este código funcione 100%, então o código criado não deve precisar de intervenção humana para rodar. Todos os comentários
devem estar devidamente dentro da sintaxe do python para não interferir ou impossibilitar no seu funcionamento,
deve estar com tratamento de erros apropriado e boas práticas do Streamlit.
Além de revisar para saber se tudo que foi pedido foi criado."
"""

def gerar_dashboard():
    """
    Função principal que gera o arquivo dashboard.py
    """
    load_dotenv()
    
    token_gemini = os.getenv("GEMINI_TOKEN")
    if not token_gemini:
        raise ValueError("Token GEMINI não encontrado no arquivo .env")
    
    genai.configure(api_key=token_gemini)
    modelo = genai.GenerativeModel("gemini-pro")
    
    codigo_visao_geral = modelo.generate_content(criar_prompt_visao_geral()).text
    codigo_abas = modelo.generate_content(criar_prompt_abas_adicionais()).text
    codigo_assistente = modelo.generate_content(criar_prompt_assistente()).text
    
    codigo_final = f"""
import streamlit as st
import pandas as pd
import yaml
import json
import plotly.express as px
from PIL import Image
import faiss
from transformers import AutoTokenizer, AutoModel
import os
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# Configuração inicial do Streamlit
st.set_page_config(
    page_title="Dashboard Câmara dos Deputados",
    layout="wide"
)

# Código da aba Visão Geral (Chain-of-Thought)
{codigo_visao_geral}

# Código das abas Despesas e Proposições (Batch Prompting)
{codigo_abas}

# Código do Assistente Virtual (Self-Ask)
{codigo_assistente}

if __name__ == "__main__":
    main()
"""
    
    with open("dashboard.py", "w", encoding="utf-8") as arquivo:
        arquivo.write(codigo_final)
    
    print("Arquivo dashboard.py criado/atualizado com sucesso!")

def main():
    try:
        logger.info("Iniciando processamento de dados...")
        
        modelo = configurar_modelo()
        if not modelo:
            logger.error("Falha ao configurar modelo. Encerrando...")
            return
        
        logger.info("Processando dados dos deputados...")
        deputados = coletar_deputados()
        if not deputados.empty:
            gerar_grafico_pizza(deputados)
            
            distribuicao_partidos = deputados["siglaPartido"].value_counts()
            prompt_distribuicao = prompt_analise_distribuicao(distribuicao_partidos)
            insights_distribuicao = gerar_resposta_modelo(modelo, prompt_distribuicao)
            
            if insights_distribuicao:
                with open("data/insights_distribuicao_deputados.json", "w", encoding="utf-8") as f:
                    json.dump({"insights": insights_distribuicao}, f, ensure_ascii=False, indent=2)
        
        logger.info("Processando dados de despesas...")
        despesas = coletar_despesas(deputados)
        if not despesas.empty:
            resultados_analise = analisar_despesas(despesas)
            if resultados_analise:
                prompt_insights = prompt_insights_despesas(resultados_analise)
                insights_despesas = gerar_resposta_modelo(modelo, prompt_insights)
                
                if insights_despesas:
                    with open("data/insights_despesas_deputados.json", "w", encoding="utf-8") as f:
                        json.dump({"insights": insights_despesas}, f, ensure_ascii=False, indent=2)
        
        logger.info("Processando proposições...")
        data_inicio = datetime(2024, 8, 1)
        data_fim = datetime(2024, 8, 30)
        proposicoes = coletar_proposicoes(data_inicio, data_fim)
        
        if not proposicoes.empty:
            sumarizacoes = []
            for _, prop in proposicoes.iterrows():
                prompt_suma = prompt_sumarizacao_proposicoes(prop["ementa"])
                sumario = gerar_resposta_modelo(modelo, prompt_suma)
                if sumario:
                    sumarizacoes.append({
                        "id": prop["id"],
                        "tema": prop["tema"],
                        "sumario": sumario
                    })
            
            if sumarizacoes:
                with open("data/sumarizacao_proposicoes.json", "w", encoding="utf-8") as f:
                    json.dump({"sumarizacoes": sumarizacoes}, f, ensure_ascii=False, indent=2)
            
            criar_base_vetorial(proposicoes)
        
        logger.info("Processamento concluído com sucesso!")
        logger.info("Gerando dashboard...")
        gerar_dashboard()
        logger.info("Processamento concluído!")
    except Exception as e:
        logger.error(f"Erro durante a execução: {str(e)}")
        raise    



if __name__ == "__main__":
    main()