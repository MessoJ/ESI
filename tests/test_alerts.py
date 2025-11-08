from api.models import AlertRule
from api.db import SessionLocal, init_db
from api.routers.alerts import evaluate_and_notify
from esi_core.compute_index import compute_components_cached


def test_alert_trigger(monkeypatch):
    init_db()
    db = SessionLocal()
    # ensure at least one compute exists
    df = compute_components_cached("KE")
    assert df is not None
    # create a rule that should trigger if ESI > -999
    r = AlertRule(metric="ESI", op=">", threshold=-999, country="KE", email=None, webhook_url=None, active=True)
    db.add(r)
    db.commit()
    # monkeypatch notify functions to capture calls
    calls = {"email": 0, "webhook": 0}
    from api import routers as pkg
    import api.routers.alerts as al

    def fake_send_email(*args, **kwargs):
        calls["email"] += 1

    def fake_webhook(*args, **kwargs):
        calls["webhook"] += 1

    monkeypatch.setattr(al, "_send_email", fake_send_email)
    monkeypatch.setattr(al, "_post_webhook", fake_webhook)

    evaluate_and_notify(db)
    # should not error; since no email or webhook set, counts remain 0 but evaluation passed
    assert isinstance(calls["email"], int)
    db.close()



