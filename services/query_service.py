import random
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from config.settings import Settings
from transformers import pipeline
import time

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

settings = Settings()

PROMPT_TEMPLATE_RESPONSE = """
Responda de forma descontraída, mantendo o respeito e valores adventistas. Seja claro e conciso:

Contexto:
{context}

Pergunta:
{question}

Resposta:
"""

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        print(f"Tempo total para {func.__name__}: {time.time() - start_time}s")
        return result
    return wrapper

@measure_time
def process_query(query_text: str, message_context: str):
    summarized_query = summarize_query(message_context, query_text)
    print(f"Resumo: {summarized_query}")

    embedding_function = OpenAIEmbeddings(api_key=settings.openai_api_key)
    db = Chroma(persist_directory=settings.chroma_path, embedding_function=embedding_function)
    results = db.similarity_search_with_relevance_scores(summarized_query, k=3)

    if len(results) == 0 or results[0][1] < 0.7:
        return {"response": get_no_match_response(), "sources": []}

    context_from_db = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    full_context = f"{context_from_db}\n\n---\n\n{message_context}"
    prompt_template_response = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_RESPONSE)
    response_prompt = prompt_template_response.format_messages(context=full_context, question=query_text)
    
    model = ChatOpenAI(api_key=settings.openai_api_key, model="gpt-3.5-turbo")
    final_response = model(response_prompt)
    
    response_text = final_response.content.strip() if hasattr(final_response, "content") else str(final_response)

    sources = [doc.metadata.get("source", "Unknown") for doc, _score in results]
    
    return {"response": response_text, "sources": sources}

#Sumarize context with query
def summarize_query(context: str, query:str):
    input_text = f"Context: {context}. Query:{query}"
    summary = summarizer(input_text, max_length=100, min_length=2, do_sample=False)
    return summary[0]['summary_text']

NO_MATCH_RESPONSES = [
    "Poxa, revirei tudo por aqui e não encontrei nada que ajude. Mas posso falar sobre química ou o livro 'Nisto Cremos' se quiser. Tente ajustar sua pergunta! 😉",
    "Hmm, parece que fiquei sem respostas desta vez. Experimente reformular sua solicitação! Sabia que química envolve desde moléculas até reações incríveis? Posso ajudar nisso também! 😊",
    "Ops! Não encontrei nada relevante dessa vez. Que tal tentar de outra forma? Também sei bastante sobre o livro 'Nisto Cremos', caso queira explorar temas de fé e doutrina! 🙏",
    "Procurei em todos os cantos, mas nada se encaixou. Sabia que química estuda transformações da matéria? Ou, se preferir, posso explicar algum ponto do 'Nisto Cremos'. Ajuste sua pergunta e seguimos! 😊",
    "Parece que fiquei sem palavras... ou melhor, sem resultados! Posso ajudar com conceitos de química ou os 28 pontos de doutrina do 'Nisto Cremos'. Que tal reformular? 😅",
    "Ainda não encontrei nada relacionado. Mas ei, sabia que o 'Nisto Cremos' aborda temas profundos como a criação e a salvação? Ou que química é a base de muitas tecnologias? Reformule sua pergunta! 😉",
    "Caramba, essa foi difícil! Não achei nada por aqui. Talvez você queira saber algo sobre reações químicas ou os fundamentos do 'Nisto Cremos'? Ajuste sua solicitação! ✨",
    "Nada ainda! Mas sabia que o livro 'Nisto Cremos' é um guia espiritual sobre crenças adventistas? Ou que química une ciência e curiosidade? Me envie outra pergunta! 🚀",
    "Essa busca me deixou no vácuo. Que tal tentar de outro jeito? Eu posso explicar os pilares da química ou detalhar pontos do 'Nisto Cremos'. É só perguntar! 😅",
    "A busca deu zero resultados, mas calma! Posso ajudar com conceitos químicos ou temas do 'Nisto Cremos', como a Trindade ou a Criação. Reformule e seguimos! 🙌"
]

# Selecionar uma resposta aleatória
def get_no_match_response():
    return random.choice(NO_MATCH_RESPONSES)