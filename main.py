from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.config import get_settings
from core.middlewears import RequestIDMiddleware
from core.redis import RedisCache
from utils.logger import logger
from api.v1.endpoints.agents_route import router as agents_router

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting up application...")
    try:
        await RedisCache().init()
        logger.info("✅ RedisCache initialized.")
    except Exception as e:
        logger.error(f"❌ Redis init failed: {e}")
        raise

    yield  # Application runs here

    logger.info("🛑 Shutting down application...")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.API_VERSION,
    debug=settings.DEBUG,
    lifespan=lifespan,
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc",  # ReDoc UI
)

app.add_middleware(
    RequestIDMiddleware,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# health check endpoint
@app.get("/health", tags=["Health Check"])
async def health_check():
    logger.info("Health check endpoint called.")
    return {"status": "ok", "version": settings.API_VERSION}

@app.get("/", tags=["Root"])
async def root():
    logger.info("Root endpoint called.")
    return {"message": "Welcome to the MediAssist", "version": settings.API_VERSION}

# Include routers for different APIs
app.include_router(
    agents_router,
    tags=["Agents"],
)