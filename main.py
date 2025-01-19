from fastapi import FastAPI
from api.routes import router
from config.settings import Settings

app = FastAPI()

settings = Settings()

app.include_router(router)

@app.get("/")
def health_check():
    return {"message": "API is running!"}
