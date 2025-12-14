import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from config.settings import Settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Starting application...")
    yield
    logging.info("Shutting down application...")


app = FastAPI(
    title="LangChain RAG API",
    description="RAG API for educational content using LangChain",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", tags=["health"])
async def health_check():
    return {"status": "healthy", "message": "API is running"}
