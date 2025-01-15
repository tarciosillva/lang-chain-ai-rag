from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
import random

# Configurar o banco de dados vetorial (Chroma)
embedding_model = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)

# Configurar o modelo de linguagem
llm = OpenAI(temperature=0.7)

# Configurar o agente de perguntas
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

def generate_quiz_question(topic, used_questions):
    # Recuperar informações relacionadas ao tópico
    docs = retriever.get_relevant_documents(topic)
    context = " ".join([doc.page_content for doc in docs])
    
    # Prompt para o LLM gerar uma pergunta e opções de resposta
    prompt = f"""
    Baseado no contexto abaixo, crie uma pergunta sobre o tópico '{topic}' com 4 opções de resposta,
    indicando qual delas é a correta. O contexto é:
    
    {context}
    
    A pergunta deve seguir este formato:
    Pergunta: ...
    A) ...
    B) ...
    C) ...
    D) ...
    Resposta correta: ...
    
    Não repita perguntas que já foram feitas. Perguntas já feitas: {used_questions}.
    """
    
    response = llm(prompt)
    
    # Extrair a pergunta e verificar duplicação
    if any(question in response for question in used_questions):
        return generate_quiz_question(topic, used_questions)
    
    return response

def run_quiz(topic, num_questions=5):
    used_questions = []
    for _ in range(num_questions):
        question = generate_quiz_question(topic, used_questions)
        print(question)
        used_questions.append(question.split("\n")[0])  # Armazena o texto da pergunta

# Exemplo de execução
run_quiz("Inteligência Artificial", num_questions=3)
