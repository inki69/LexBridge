from contextlib import asynccontextmanager
from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient

from helpers.config import get_settings
from stores.llm.factory import LLMFactory
from stores.vectordb.provider import QdrantProvider
from controllers import DataController, RAGController
from routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    # MongoDB
    mongo_client = AsyncIOMotorClient(settings.MONGODB_URI)
    db = mongo_client[settings.MONGODB_DB_NAME]

    # Vector store (local path — no separate Docker service needed)
    vector_store = QdrantProvider(
        path=settings.VECTOR_DB_PATH,
        distance_metric=settings.VECTOR_DISTANCE_METRIC,
    )

    # LLM provider via Factory Pattern
    # Swap between Gemini / OpenAI / Ollama by changing LLM_BACKEND in .env
    llm = LLMFactory.create(settings)

    app.state.db = db
    app.state.data_controller = DataController(db=db, vector_store=vector_store, llm=llm)
    app.state.rag_controller = RAGController(db=db, vector_store=vector_store, llm=llm)

    yield

    mongo_client.close()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
