
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
import google.generativeai as genai

# Carrega variáveis de ambiente
load_dotenv()

GEMINI_TOKEN = os.getenv("GEMINI_TOKEN")
# Configuração inicial do Streamlit
#st.set_page_config(
#    page_title="Dashboard Câmara dos Deputados",
#    layout="wide"
#)

# Código da aba Visão Geral (Chain-of-Thought)

import streamlit as st
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(
    page_title="Dashboard Parlamentar",
    page_icon="📊",
    layout="wide"
)

# Etapa 1 - Análise Inicial
# Bibliotecas
# - streamlit
# - yaml
# - json
# - numpy
# - matplotlib

# Estrutura básica
st.sidebar.title("Configurações")
st.sidebar.markdown("---")
aba = st.sidebar.selectbox("Selecione uma aba", ["Overview", "Distribuição de Deputados", "Despesas", "Proposições"])

# Etapa 2 - Desenvolvimento da Aba Visão Geral

# Título e descrição
if aba == "Overview":
    st.title("Overview do Parlamento")
    st.markdown("Este dashboard fornece uma visão geral do Parlamento Brasileiro, com informações sobre a distribuição de deputados por partido, região e gênero.")

    # Carregando config.yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    st.markdown("### Resumo")
    st.markdown(config['overview_summary'])
    # Gráfico de barras
    image = Image.open("docs/distribuicao_deputados.png")
    st.image(image, caption="Distribuição de Deputados por Partido", use_column_width=True)

    # Insights
    with open("data/insights_distribuicao_deputados.json", "r") as f:
        insights = json.load(f)

    st.markdown(f"**Insight 1:** {insights['insight1']}")
    st.markdown(f"**Insight 2:** {insights['insight2']}")

# Etapa 3 - Implementação Final
# Tratamento de erros
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    with open("data/insights_distribuicao_deputados.json", "r") as f:
        insights = json.load(f)
except FileNotFoundError as e:
    st.error(f"Arquivo não encontrado: {e}")
except Exception as e:
    st.error(f"Erro desconhecido: {e}")

# Comentários e organização
# Os comentários estão devidamente dentro da sintaxe do Python
# O código está organizado com indentação e espaçamento adequados


# Código das abas Despesas e Proposições (Batch Prompting)

import pandas as pd
import altair as alt
import streamlit as st
import json

# Carregar dados
with open('data/insights_despesas_deputados.json') as f:
    insights_despesas = json.load(f)
df_despesas = pd.read_parquet('data/serie_despesas_diarias_deputados.parquet')
df_proposicoes = pd.read_parquet('data/proposicoes_deputados.parquet')
df_deputados = pd.read_parquet('data/deputados.parquet')

with open('data/sumarizacao_proposicoes.json') as f:
        sumarizacao_proposicoes = json.load(f)
with open('data/insights_distribuicao_deputados.json') as f:
        distribuicao_deputados = json.load(f)        

if aba == "Despesas":
    
    # Aba Despesas
    st.header('Despesas')

    # Exibir insights
    st.subheader('Insights sobre Despesas')
    st.write(insights_despesas)

    # Caixa de seleção para escolha do deputado
    deputados = df_despesas['nome'].unique()
    deputado_selecionado = st.selectbox('Deputado:', deputados)

    # Gráfico de barras temporal
    df_despesas_deputado = df_despesas[df_despesas['nome'] == deputado_selecionado]
    chart = alt.Chart(df_despesas_deputado).mark_bar().encode(
        x='data',
        y='valor',
        color='categoria'
    )
    st.write(chart)

# Aba Proposições
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
model = AutoModel.from_pretrained("neuralmind/bert-base-portuguese-cased")

# Carregar a base de dados FAISS
index = faiss.read_index("data/proposicoes_index.faiss")
vectors = pd.read_parquet('data/proposicoes_vetorizadas.parquet').values

# Função para realizar busca no FAISS
def busca_faiss(pergunta):
    """Realiza busca semântica no FAISS"""
    try:
        tokens_pergunta = tokenizer(
            pergunta, 
            return_tensors="np", 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        vetor_pergunta = model(**tokens_pergunta).last_hidden_state.mean(axis=1).squeeze(0).numpy()
        distancias, indices = index.search(np.array([vetor_pergunta]), k=5)
        return distancias, indices
    except Exception as erro:
        print(f"Erro na busca FAISS: {erro}")
        return None, None

# Função para analisar resposta
def analisar_resposta(distancias, indices):
    """Analisa resultados da busca FAISS"""
    try:
        if len(distancias) > 0 and distancias[0][0] < 0.5:
            indice_resultado = indices[0][0]
            return vectors[indice_resultado]
        return "Nenhuma informação relevante encontrada."
    except Exception as erro:
        print(f"Erro na análise da resposta: {erro}")
        return None

def configurar_modelo_ia():
    """Configura o modelo Gemini Pro"""
    try:
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
def gerar_contexto(pergunta):
    """Gera contexto combinando dados relevantes"""
    distancias, indices = busca_faiss(pergunta)
    if indices is not None:
        proposicoes_relevantes = df_proposicoes.iloc[indices[0]]
        
        contexto = f"""
        Dados das Proposições Relevantes:
        {proposicoes_relevantes}
        
        Resumo das Proposições:
        {sumarizacao_proposicoes}
        
        Informações distribuição dos Deputados:
        {distribuicao_deputados}
        
        Despesas dos Deputados:
        {insights_despesas}

        Tabela Deputados:
        {df_deputados}

        Tabela despesas dos Deputados:
        {df_despesas}
        """
        return contexto
    else:
        contexto = f"""

        
        Resumo das Proposições:
        {sumarizacao_proposicoes}
        
        Informações dos Deputados:
        {distribuicao_deputados}
        
        Despesas dos Deputados:
        {insights_despesas}

        Tabela Deputados:
        {df_deputados}

        Tabela despesas dos Deputados:
        {df_despesas}
        """
        return contexto
    
    #return ""
def criar_prompt_self_ask(pergunta, contexto):
    """Cria prompt usando técnica Self-Ask"""
    return f"""Você é um assistente especialista em dados da Câmara dos Deputados.
    
    Para responder à pergunta, siga estas etapas:
    1. Primeiro, identifique qual informação específica está sendo solicitada
    2. Depois, verifique quais dados do contexto são relevantes
    3. Analise os dados relevantes para encontrar a resposta
    4. Por fim, formule uma resposta clara e objetiva
    
    Pergunta do usuário: {pergunta}
    
    Contexto disponível:
    {contexto}
    
    Resposta passo a passo:"""

def interface_proposicoes():
    """Interface principal da aba Proposições"""
    st.header("Proposições")

    # Exibição dos dados básicos
    st.subheader("Proposições por Deputado")
    st.dataframe(df_proposicoes)

    # Filtros
    st.subheader("Filtrar Proposições")
    colunas = df_proposicoes.columns.tolist()
    coluna_selecionada = st.selectbox("Selecione a coluna para filtrar:", colunas)

    if coluna_selecionada:
        valores = df_proposicoes[coluna_selecionada].dropna().unique().tolist()
        valor_selecionado = st.selectbox(f"Selecione o valor para filtrar em {coluna_selecionada}:", valores)
        df_filtrado = df_proposicoes[df_proposicoes[coluna_selecionada] == valor_selecionado]
        st.dataframe(df_filtrado)

    # Assistente Virtual
    st.subheader("Assistente Virtual para Análise de Dados da Câmara dos Deputados")
    st.markdown("Faça sua pergunta sobre proposições, despesas e deputados.")

    # Chat
    if "mensagens" not in st.session_state:
        st.session_state.mensagens = []

    pergunta = st.text_input("Digite sua pergunta:", key="campo_pergunta")

    if st.button("Enviar"):
        if pergunta:
            # Adiciona pergunta do usuário ao histórico
            st.session_state.mensagens.append({"role": "user", "content": pergunta})

            try:
                # Gera resposta
                contexto = gerar_contexto(pergunta)
                prompt = criar_prompt_self_ask(pergunta, contexto)
                modelo = configurar_modelo_ia()
                
                if modelo:
                    resposta = modelo.generate_content(prompt)
                    texto_resposta = resposta.text
                    
                    # Adiciona resposta ao histórico
                    st.session_state.mensagens.append({"role": "assistant", "content": texto_resposta})
                else:
                    st.error("Erro ao configurar o modelo de IA.")
            except Exception as erro:
                st.error(f"Erro ao processar a pergunta: {erro}")

    # Exibe histórico do chat
    for msg in st.session_state.mensagens:
        if msg["role"] == "user":
            st.write("Você:", msg["content"])
        else:
            st.write("Assistente:", msg["content"]) 

    st.subheader("Resumo das Proposições")
    st.write(sumarizacao_proposicoes)            

    


if __name__ == "__main__":
    interface_proposicoes()