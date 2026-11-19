from __future__ import annotations
from pathlib import Path
from fastapi.testclient import TestClient
from findex.index import build_inverted_index
from findex.storage import save_index
from findex.web.app import create_app

def _prepare(tmp_path: Path, monkeypatch) -> None:
    c = tmp_path / 'corpus'
    c.mkdir()
    (c / 'a.txt').write_text('Python Search Engine\nInverted index with BM25 ranking.', encoding='utf-8')
    (c / 'b.txt').write_text('FastAPI Service\nWeb API running on Uvicorn.', encoding='utf-8')
    idx_path = tmp_path / 'index.bin'
    idx = build_inverted_index(c, include_positions=True)
    save_index(idx, idx_path, fmt='bin')
    idx.close()
    monkeypatch.setenv('INDEX_PATH', str(idx_path))
    monkeypatch.setenv('CORPUS_DIR', str(c))

def test_missing_index(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv('INDEX_PATH', str(tmp_path / 'none.bin'))
    monkeypatch.setenv('CORPUS_DIR', str(tmp_path / 'none'))
    with TestClient(create_app()) as client:
        assert client.get('/health').json()['ready'] is False

def test_search_and_ui(tmp_path: Path, monkeypatch) -> None:
    _prepare(tmp_path, monkeypatch)
    with TestClient(create_app()) as client:
        assert client.get('/health').json()['ready'] is True
        assert client.get('/stats').json()['corpus_files'] == 2
        r = client.get('/api/search', params={'q': 'BM25', 'scorer': 'bm25'}).json()
        assert r['total'] == 1
        r2 = client.post('/api/search', json={'query': 'FastAPI', 'scorer': 'tfidf'}).json()
        assert r2['total'] == 1
        doc_id = r['results'][0]['doc_id']
        assert 'BM25' in client.get(f'/api/doc/{doc_id}').json()['content']
        assert 'Python Search Engine' in client.get('/', params={'q': 'Python'}).text
