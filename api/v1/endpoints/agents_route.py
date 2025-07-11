from fastapi import APIRouter, HTTPException, Request

from models.api_models import AgentRequest, AgentResponse
from services.agent_service import run_agent_logic
from utils.logger import logger

router = APIRouter()

@router.post("/run", response_model=AgentResponse)
async def run_agent(request: Request, payload: AgentRequest):
    try:
        logger.info(f"Received agent request with session_id={payload.session_id}")
        response = await run_agent_logic(payload)

        if response.status_code >= 400:
            raise HTTPException(status_code=response.status_code, detail=response.error)

        return response

    except Exception as e:
        logger.exception("Unhandled exception in /run endpoint")
        raise