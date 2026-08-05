"""Security-focused tests for the localhost Flask dashboard."""

from __future__ import annotations

import pytest

from ai_rganize.gui.app import create_app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Keep profile/ledger lookups out of the real home dir during tests.
    monkeypatch.setenv("HOME", str(tmp_path))
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_api_organize_rejects_path_escape(client, tmp_path):
    outside = tmp_path.parent / "outside_escape"
    # Request a path that normalizes outside HOME
    resp = client.post(
        "/api/organize",
        json={"directory": str(outside), "dry_run": True},
        headers={"Host": "127.0.0.1"},
    )
    assert resp.status_code == 400
    assert "Invalid" in resp.get_json()["error"] or "disallowed" in resp.get_json()["error"]


def test_api_organize_accepts_path_under_home(client, tmp_path):
    target = tmp_path / "Downloads"
    target.mkdir()
    (target / "a.txt").write_text("hi")
    resp = client.post(
        "/api/organize",
        json={"directory": str(target), "dry_run": True},
        headers={"Host": "127.0.0.1"},
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["executed"] is False
    assert body["dry_run"] is True


def test_api_organize_hides_exception_details(client, tmp_path):
    resp = client.post(
        "/api/organize",
        json={"directory": str(tmp_path), "profile": "does-not-exist", "dry_run": True},
        headers={"Host": "127.0.0.1"},
    )
    assert resp.status_code == 404
    assert resp.get_json()["error"] == "Profile not found"


def test_non_localhost_host_rejected(client):
    resp = client.get("/api/status", headers={"Host": "evil.example"})
    assert resp.status_code == 403
