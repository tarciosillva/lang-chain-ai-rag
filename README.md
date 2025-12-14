# LangChain RAG API

API REST para sistema de RAG (Retrieval-Augmented Generation) usando LangChain, desenvolvida para auxiliar alunos e professores da Escola Adventista com tutoria de sala de aula.

## Características

- **RAG com LangChain**: Sistema de recuperação e geração aumentada usando ChromaDB
- **API REST com FastAPI**: Endpoints assíncronos e documentação automática
- **Suporte a voz**: Processamento de consultas por áudio com síntese de voz
- **Integração com Google Cloud**: Armazenamento de áudio e serviços de TTS/STT
- **Type hints completos**: Código tipado seguindo melhores práticas Python
- **Logging estruturado**: Sistema de logs para monitoramento e debug

## Pré-requisitos

- Python 3.10+
- OpenAI API Key
- Google Cloud Platform account com:
  - Cloud Storage bucket configurado
  - Service Account com permissões adequadas
  - Credenciais JSON baixadas

## Instalação

### 1. Clone o repositório

```bash
git clone <repository-url>
cd langchain-rag
```

### 2. Instale as dependências

**Nota**: Para usuários macOS, pode ser necessário instalar `onnxruntime` via conda primeiro:

```bash
conda install onnxruntime -c conda-forge
```

Para usuários Windows, instale o Microsoft C++ Build Tools seguindo [este guia](https://github.com/bycloudai/InstallVSBuildToolsWindows).

Depois, instale as dependências:

```bash
pip install -r requirements.txt
pip install "unstructured[md]"
```

### 3. Configure as variáveis de ambiente

Crie um arquivo `.env` na raiz do projeto:

```env
OPENAI_API_KEY=your_openai_api_key_here
CHROMA_PATH=chroma
GOOGLE_CREDENTIALS_PATH=path/to/your/google-credentials.json
GCP_BUCKET_NAME=your-gcp-bucket-name
```

### 4. Prepare o banco de dados vetorial

Coloque seus documentos PDF na pasta `data/books/` e execute:

```bash
python create_database.py
```

Este script irá:
- Carregar documentos PDF da pasta `data/books/`
- Dividir os documentos em chunks
- Criar embeddings usando OpenAI
- Armazenar no ChromaDB

## Executando a API

### Modo desenvolvimento

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Com Docker

```bash
docker-compose up --build
```

A API estará disponível em `http://localhost:8000`

## Documentação da API

Após iniciar o servidor, acesse:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Endpoints

### Health Check

```http
GET /
```

Retorna o status da API.

### Text Query

```http
POST /api/v1/query
Content-Type: application/json

{
  "query_text": "O que é química?",
  "message_context": ""
}
```

**Resposta:**

```json
{
  "response": "Química é a ciência que estuda...",
  "sources": ["data/books/apostila_quimica.pdf"]
}
```

### Voice Query

```http
POST /api/v1/voice-query
Content-Type: application/json

{
  "audio_url": "https://example.com/audio.ogg",
  "audio_auth": "bearer_token_here",
  "message_context": ""
}
```

**Resposta:**

```json
{
  "query_text": "Transcrição do áudio",
  "response": "Resposta gerada pelo modelo",
  "audio_link": "https://storage.googleapis.com/bucket/audio.mp3",
  "sources": ["data/books/apostila_quimica.pdf"]
}
```

## Estrutura do Projeto

```
langchain-rag/
├── api/
│   ├── __init__.py
│   └── routes.py              # Rotas da API
├── config/
│   ├── __init__.py
│   └── settings.py           # Configurações e variáveis de ambiente
├── models/
│   ├── __init__.py
│   ├── request_models.py     # Modelos Pydantic para requests
│   └── response_models.py    # Modelos Pydantic para responses
├── services/
│   ├── __init__.py
│   ├── query_service.py      # Serviço de processamento de queries
│   └── voice_query_service.py # Serviço de processamento de voz
├── data/
│   └── books/                # Documentos PDF para indexação
├── chroma/                   # Banco de dados vetorial (gerado)
├── main.py                   # Aplicação FastAPI principal
├── create_database.py        # Script para criar o banco vetorial
├── requirements.txt          # Dependências Python
├── Dockerfile               # Configuração Docker
├── docker-compose.yaml      # Orquestração Docker
└── README.md                # Este arquivo
```

## Tecnologias Utilizadas

- **FastAPI**: Framework web assíncrono
- **LangChain**: Framework para aplicações LLM
- **ChromaDB**: Banco de dados vetorial
- **OpenAI**: Embeddings e modelo de linguagem
- **Google Cloud**: Storage e serviços de voz
- **Pydantic**: Validação de dados
- **Transformers**: Modelo de sumarização (BART)

## Desenvolvimento

### Formatação e Linting

O projeto segue as melhores práticas Python:

- Type hints em todas as funções
- Validação com Pydantic
- Logging estruturado
- Tratamento de exceções adequado
- Código assíncrono onde apropriado

### Testando os Endpoints

Você pode testar os endpoints usando:

1. **Swagger UI**: Acesse `http://localhost:8000/docs`
2. **curl**:

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"query_text": "O que é química?", "message_context": ""}'
```

3. **Python requests**:

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/query",
    json={
        "query_text": "O que é química?",
        "message_context": ""
    }
)
print(response.json())
```

## Troubleshooting

### Erro ao instalar dependências

- **macOS**: Instale `onnxruntime` via conda primeiro
- **Windows**: Instale Microsoft C++ Build Tools
- **Linux**: Certifique-se de ter `build-essential` instalado

### Erro ao criar banco de dados

- Verifique se os PDFs estão na pasta `data/books/`
- Confirme que a `OPENAI_API_KEY` está configurada
- Verifique os logs para mais detalhes

### Erro ao processar queries de voz

- Verifique se as credenciais do Google Cloud estão corretas
- Confirme que o bucket do GCP existe e está acessível
- Verifique se o arquivo de credenciais está no caminho correto

## Licença

Este projeto foi desenvolvido em parceria entre o professor Edmar e o desenvolvedor Tarcio para a Escola Adventista.

## Suporte

Para questões ou problemas, abra uma issue no repositório do projeto.