"""FastAPI product assistant with a SQLite-persisted LangGraph workflow."""

from contextlib import asynccontextmanager
from functools import lru_cache
import hashlib
import logging
import os
from pathlib import Path
import re
import secrets
from threading import Lock

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from customer_agent import ProductAgent
from shopping_graph import build_shopping_graph


load_dotenv()
log = logging.getLogger(__name__)
WEB = Path(__file__).parent / "web"
COOKIE_NAME = "shopping_session"
SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{32,128}$")
# SQLite has one writer; also prevent two confirmations in this process racing the same thread.
_graph_lock = Lock()


@lru_cache(maxsize=1)
def get_agent() -> ProductAgent:
    return ProductAgent()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configured = Path(os.getenv("CHECKPOINT_SQLITE_PATH", "var/checkpoints.sqlite3"))
    path = configured if configured.is_absolute() else Path(__file__).parent / configured
    path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(path)) as saver:
        app.state.graph = build_shopping_graph(get_agent(), saver)
        yield


app = FastAPI(title="商品智能客服", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB), name="static")


class Turn(BaseModel):
    role: str
    content: str = Field(min_length=1, max_length=2000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    history: list[Turn] = Field(default_factory=list, max_length=8)


class RefundDecision(BaseModel):
    approved: bool


def get_graph(request: Request):
    return request.app.state.graph


def session_config(token: str) -> dict:
    # A random browser cookie is the demo identity. Do not accept a user_id from JSON.
    return {"configurable": {"thread_id": hashlib.sha256(token.encode("ascii")).hexdigest()}}


def session_token(request: Request, response: Response, *, create: bool) -> str | None:
    token = request.cookies.get(COOKIE_NAME)
    if not token or not SESSION_RE.fullmatch(token):
        if not create:
            return None
        token = secrets.token_urlsafe(32)
        response.set_cookie(COOKIE_NAME, token, httponly=True, secure=request.url.scheme == "https", samesite="lax", max_age=30 * 24 * 3600)
    return token


def output(state: dict, *, pending: bool) -> dict:
    return {
        "answer": state.get("answer", ""), "products": state.get("products", []),
        "orders": state.get("orders", []), "pending_refund": pending,
        "quote": state.get("quote") if pending else None,
        "result": state.get("result"),
    }


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(WEB / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/session")
def current_session(request: Request, response: Response, graph=Depends(get_graph)):
    token = session_token(request, response, create=False)
    if token is None:
        return output({}, pending=False)
    with _graph_lock:
        snapshot = graph.get_state(session_config(token))
    return output(snapshot.values, pending="confirm" in snapshot.next)


@app.post("/api/chat")
def chat(body: ChatRequest, request: Request, response: Response, graph=Depends(get_graph)):
    token = session_token(request, response, create=True)
    config = session_config(token)
    try:
        with _graph_lock:
            previous = graph.get_state(config)
            if "confirm" in previous.next:
                raise HTTPException(status_code=409, detail="请先确认或取消当前的模拟退款")
            saved = previous.values
            history = list(saved.get("history") or [])
            if saved.get("question") and saved.get("answer"):
                history.extend((
                    {"role": "user", "content": saved["question"]},
                    {"role": "assistant", "content": saved["answer"]},
                ))
            history = history[-8:]
            graph.invoke({
                "question": body.message.strip(), "history": history, "products": [],
                "quote": None, "result": None, "answer": "", "approved": False,
                "wants_order": False, "wants_refund": False,
            }, config=config)
            snapshot = graph.get_state(config)
        return output(snapshot.values, pending="confirm" in snapshot.next)
    except HTTPException:
        raise
    except Exception:
        log.exception("商品客服回答失败")
        raise HTTPException(status_code=503, detail="商品服务暂时不可用，请稍后重试或检查服务配置")


@app.post("/api/refund/confirm")
def confirm_refund(body: RefundDecision, request: Request, response: Response, graph=Depends(get_graph)):
    token = session_token(request, response, create=False)
    if token is None:
        raise HTTPException(status_code=409, detail="当前会话没有待确认的模拟退款")
    config = session_config(token)
    try:
        with _graph_lock:
            snapshot = graph.get_state(config)
            if "confirm" not in snapshot.next or not snapshot.values.get("quote"):
                raise HTTPException(status_code=409, detail="当前会话没有待确认的模拟退款")
            graph.invoke(Command(resume={"approved": body.approved}), config=config)
            final = graph.get_state(config)
        return output(final.values, pending="confirm" in final.next)
    except HTTPException:
        raise
    except Exception:
        log.exception("模拟退款确认失败")
        raise HTTPException(status_code=503, detail="模拟退款操作失败，请查询当前会话状态")
