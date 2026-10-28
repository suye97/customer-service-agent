"""Product shopping assistant API and web interface."""

from functools import lru_cache
from pathlib import Path
import logging

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from customer_agent import ProductAgent


load_dotenv()
log = logging.getLogger(__name__)
app = FastAPI(title="商品智能客服")
WEB = Path(__file__).parent / "web"
app.mount("/static", StaticFiles(directory=WEB), name="static")


class Turn(BaseModel):
    role: str
    content: str = Field(min_length=1, max_length=2000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    history: list[Turn] = Field(default_factory=list, max_length=8)


@lru_cache(maxsize=1)
def get_agent() -> ProductAgent:
    return ProductAgent()


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(WEB / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat")
def chat(request: ChatRequest, agent: ProductAgent = Depends(get_agent)):
    history = [turn.model_dump() for turn in request.history if turn.role in ("user", "assistant")]
    try:
        return agent.answer(request.message.strip(), history)
    except Exception:
        log.exception("商品客服回答失败")
        raise HTTPException(status_code=503, detail="商品服务暂时不可用，请稍后重试或检查服务配置")
