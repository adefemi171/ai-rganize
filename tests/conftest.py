"""Shared pytest fixtures. Isolates all AI-rganize state under a tmp HOME
so tests never read/write the real user's ~/.ai_rganize directory."""

from __future__ import annotations

import pytest


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Redirect every module-level 'home' path constant to a tmp directory."""
    import ai_rganize.config.profile as profile_mod
    import ai_rganize.memory.feedback as feedback_mod
    import ai_rganize.provenance.ledger as ledger_mod

    fake_home = tmp_path / "home"
    fake_home.mkdir()

    profiles_dir = fake_home / ".ai_rganize" / "profiles"
    monkeypatch.setattr(profile_mod, "PROFILE_HOME", fake_home / ".ai_rganize")
    monkeypatch.setattr(profile_mod, "PROFILES_DIR", profiles_dir)

    ledger_path = fake_home / ".ai_rganize" / "ledger.jsonl"
    monkeypatch.setattr(ledger_mod, "LEDGER_HOME", fake_home / ".ai_rganize")
    monkeypatch.setattr(ledger_mod, "LEDGER_PATH", ledger_path)

    memory_home = fake_home / ".ai_rganize" / "memory"
    monkeypatch.setattr(feedback_mod, "MEMORY_HOME", memory_home)

    return fake_home
