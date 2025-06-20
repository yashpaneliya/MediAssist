import uuid
from models.api_models import AgentRequest, AgentResponse

from utils.logger import logger

async def run_agent_logic(payload: AgentRequest) -> AgentResponse:
    logger.debug(f"Running agent with query='{payload.query}', session_id='{payload.session_id}'")
    try:
        session_id = payload.session_id or str(uuid.uuid4())

        # TODO: fetch history from cache if required based on session_id

        # TODO: call the agent and process the query

        result = f"{payload.query}"

        return AgentResponse(
            response=result,
            session_id=session_id,
            status_code=200,
        )

    except Exception as e:
        logger.exception("Agent processing failed")
        return AgentResponse(
            response="",
            session_id=payload.session_id or "unknown",
            error=str(e),
            status_code=500,
        )
