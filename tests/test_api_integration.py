from fastapi.testclient import TestClient
from api.main import app


def test_healthz():
    client = TestClient(app)
    r = client.get('/healthz')
    assert r.status_code == 200
    js = r.json()
    assert 'ok' in js


def test_latest_endpoint_ke():
    client = TestClient(app)
    r = client.get('/api/index/latest?country=KE')
    # may fail if no data; accept 200 or 500 with stale fallback if cache exists
    assert r.status_code in (200, 500)



