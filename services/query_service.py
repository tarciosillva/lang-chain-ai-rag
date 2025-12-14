import logging
import random
from typing import Any, Dict

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from transformers import pipeline

from config.settings import Settings

logger = logging.getLogger(__name__)

settings = Settings()

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """Seu nome é Lia. Você é uma assistente virtual de tutoria de sala de aula da Escola Adventista, criada para auxiliar alunos e professores.  
Você foi desenvolvida em parceria entre o professor Edmar e o desenvolvedor de software Tarcio, sendo lançada em janeiro de 2025, atualmente na versão beta.
Seu objetivo é responder perguntas relacionadas ao contexto escolar em sala de aula, incluindo dúvidas sobre disciplinas de forma descontraída, clara e direta.

Use o seguinte contexto para responder a pergunta. Se não encontrar informações relevantes no contexto, seja honesto sobre isso.

Contexto:
{context}

Histórico da conversa:
{message_context}"""),
    ("human", "{input}")
])

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

_summarizer = None


def _get_summarizer() -> pipeline:
    global _summarizer
    if _summarizer is None:
        _summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return _summarizer


def _get_vector_store() -> Chroma:
    embedding_function = OpenAIEmbeddings(api_key=settings.openai_api_key)
    return Chroma(
        persist_directory=settings.chroma_path_str,
        embedding_function=embedding_function
    )


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.openai_api_key,
        model="gpt-3.5-turbo",
        temperature=0.7
    )


def _summarize_query(context: str, query: str) -> str:
    try:
        summarizer = _get_summarizer()
        input_text = f"Context: {context}. Query: {query}"
        summary = summarizer(input_text, max_length=100, min_length=2, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        logger.warning(f"Error summarizing query: {e}. Using original query.")
        return query


def _get_no_match_response() -> str:
    return random.choice(NO_MATCH_RESPONSES)


def process_query(query_text: str, message_context: str) -> Dict[str, Any]:
    try:
        summarized_query = _summarize_query(message_context, query_text)
        
        vector_store = _get_vector_store()
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.7}
        )
        
        results = retriever.get_relevant_documents(summarized_query)
        
        if not results:
            return {
                "response": _get_no_match_response(),
                "sources": []
            }
        
        llm = _get_llm()
        
        prompt_with_context = PROMPT_TEMPLATE.partial(message_context=message_context)
        document_chain = create_stuff_documents_chain(llm, prompt_with_context)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        response = retrieval_chain.invoke({
            "input": query_text
        })
        
        sources = [
            doc.metadata.get("source", "Unknown")
            for doc in results
        ]
        
        return {
            "response": response["answer"].strip(),
            "sources": sources
        }
        
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        return {
            "response": "Desculpe, ocorreu um erro ao processar sua pergunta. Por favor, tente novamente.",
            "sources": []
        }