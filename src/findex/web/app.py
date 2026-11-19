from __future__ import annotations
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from findex.web.adapter import FindexEngineAdapter
from findex.web.config import get_settings

TEMPLATES_DIR = Path(__file__).resolve().parent / 'templates'

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=100)
    scorer: Literal['bm25', 'tfidf'] = Field(default='bm25')

class SearchResultItem(BaseModel):
    doc_id: str
    title: str
    score: float
    snippet: str
    path: str | None = None

class SearchResponse(BaseModel):
    query: str
    scorer: str
    total: int
    took_ms: float
    results: list[SearchResultItem]

def create_app() -> FastAPI:
    settings = get_settings()
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = FindexEngineAdapter(index_path=settings.index_path, corpus_dir=settings.corpus_dir)
        app.state.settings = settings
        app.state.engine = engine
        try:
            yield
        finally:
            engine.close()

    app = FastAPI(title='findex Search API', version='0.1.0', lifespan=lifespan)
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    def _get_engine(request: Request) -> FindexEngineAdapter:
        engine = getattr(request.app.state, 'engine', None)
        if engine is None:
            engine = FindexEngineAdapter(settings.index_path, settings.corpus_dir)
            request.app.state.engine = engine
        return engine

    @app.get('/', response_class=HTMLResponse, include_in_schema=False)
    async def index_page(request: Request, q: str = Query(default=''), top_k: int = Query(default=10, ge=1, le=100), scorer: Literal['bm25', 'tfidf'] = Query(default='bm25')) -> HTMLResponse:
        engine = _get_engine(request)
        hits, took_ms = engine.search(q, top_k=top_k, scorer_name=scorer) if q.strip() else ([], 0.0)
        return templates.TemplateResponse(request, 'index.html', {'query': q, 'top_k': top_k, 'scorer': scorer, 'results': hits, 'took_ms': took_ms, 'stats': engine.stats()})

    @app.get('/health')
    async def health(request: Request) -> dict[str, Any]:
        engine = _get_engine(request)
        st = engine.stats()
        return {'status': st['status'], 'ready': engine.is_ready, 'engine': st['engine']}

    @app.get('/stats')
    async def stats(request: Request) -> dict[str, Any]:
        return _get_engine(request).stats()

    @app.get('/api/search', response_model=SearchResponse)
    async def api_search_get(request: Request, q: str = Query(..., min_length=1, max_length=1000), top_k: int = Query(default=10, ge=1, le=100), scorer: Literal['bm25', 'tfidf'] = Query(default='bm25')) -> SearchResponse:
        hits, took_ms = _get_engine(request).search(q, top_k=top_k, scorer_name=scorer)
        return SearchResponse(query=q, scorer=scorer, total=len(hits), took_ms=took_ms, results=[SearchResultItem(doc_id=h.doc_id, title=h.title, score=h.score, snippet=h.snippet, path=h.path) for h in hits])

    @app.post('/api/search', response_model=SearchResponse)
    async def api_search_post(request: Request, payload: SearchRequest) -> SearchResponse:
        hits, took_ms = _get_engine(request).search(payload.query, top_k=payload.top_k, scorer_name=payload.scorer)
        return SearchResponse(query=payload.query, scorer=payload.scorer, total=len(hits), took_ms=took_ms, results=[SearchResultItem(doc_id=h.doc_id, title=h.title, score=h.score, snippet=h.snippet, path=h.path) for h in hits])

    @app.get('/api/doc/{doc_id:path}')
    async def api_get_document(request: Request, doc_id: str) -> dict[str, Any]:
        doc = _get_engine(request).get_document(doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail=f'Документ {doc_id} не знайдено')
        return doc

    return app
