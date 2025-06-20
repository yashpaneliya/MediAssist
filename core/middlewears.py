import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from utils.logger import request_id_ctx_var

class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware to generate and attach a unique request ID per request.
    The ID is stored in a context variable and used in all logs.
    """
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())  # or use shorter: uuid.uuid4().hex[:8]
        request_id_ctx_var.set(request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id  # Optional: send back to client
        return response
