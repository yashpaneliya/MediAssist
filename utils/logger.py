from loguru import logger
import sys
import contextvars

from core.config import get_settings

request_id_ctx_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

# Custom function to inject request_id into each log
def request_id_filter(record):
    record["extra"]["request_id"] = request_id_ctx_var.get("-")
    return True

settings = get_settings()

logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    level="DEBUG" if settings.DEBUG else "INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
           "REQ_ID=<yellow>{extra[request_id]}</yellow> - "
           "<level>{message}</level>",
    filter=request_id_filter,
    enqueue=True,
    backtrace=True,
    diagnose=settings.DEBUG,
)
