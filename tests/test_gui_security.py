"""Security-focused tests for the localhost Flask dashboard."""

from __future__ import annotations

import pytest

from ai_rganize.gui.app import create_app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Keep profile/ledger lookups out of the real home dir during tests.
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "Downloads").mkdir()
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_api_organize_rejects_raw_filesystem_path(client):
    resp = client.post(
        "/api/organize",
        json={"directory": "/etc", "dry_run": True},
        headers={"Host": "127.0.0.1"},
    )
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Invalid target"


def test_api_organize_accepts_named_target(client):
    resp = client.post(
        "/api/organize",
        json={"directory": "Downloads", "dry_run": True},
        headers={"Host": "127.0.0.1"},
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["executed"] is False
    assert body["dry_run"] is True
    assert body["target"] == "Downloads"


def test_api_organize_hides_exception_details(client, tmp_path):
    resp = client.post(
        "/api/organize",
        json={"directory": "Downloads", "profile": "does-not-exist", "dry_run": True},
        headers={"Host": "127.0.0.1"},
    )
    assert resp.status_code == 404
    assert resp.get_json()["error"] == "Profile not found"


def test_non_localhost_host_rejected(client):
    resp = client.get("/api/status", headers={"Host": "evil.example"})
    assert resp.status_code == 403
