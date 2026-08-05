"""Tests for ai_rganize.council.LLMCouncil, with all provider calls mocked
out (no network access)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ai_rganize.council.council import LLMCouncil


class _FakeClient:
    def __init__(self, folders):
        self._folders = folders

    def categorize_files(self, file_batch, verbose=False, max_folders=None, **kwargs):
        return self._folders


class _FailingClient:
    def categorize_files(self, *args, **kwargs):
        raise RuntimeError("simulated provider outage")


def _files(n):
    return [{"name": f"file_{i}.txt", "path": f"/tmp/file_{i}.txt"} for i in range(n)]


def test_council_requires_at_least_one_provider():
    with pytest.raises(ValueError):
        LLMCouncil(providers=[])


def test_council_majority_vote_agreement(monkeypatch):
    files = _files(2)

    clients = {
        "openai": _FakeClient(["Documents", "Images"]),
        "claude": _FakeClient(["Documents", "Images"]),
        "gemini": _FakeClient(["Finance", "Images"]),
    }

    def fake_create(provider, api_key=None, model=None):
        return clients[provider]

    with patch("ai_rganize.council.council.create_ai_client", side_effect=fake_create):
        council = LLMCouncil(providers=["openai", "claude", "gemini"])
        folders, confidences = council.categorize(files)

    assert folders[0] == "Documents"  # 2 of 3 voted Documents
    assert confidences[0] == pytest.approx(2 / 3)

    assert folders[1] == "Images"  # unanimous
    assert confidences[1] == pytest.approx(1.0)


def test_council_skips_failing_providers(monkeypatch):
    files = _files(1)

    clients = {
        "openai": _FakeClient(["Documents"]),
        "broken": _FailingClient(),
    }

    def fake_create(provider, api_key=None, model=None):
        return clients[provider]

    with patch("ai_rganize.council.council.create_ai_client", side_effect=fake_create):
        council = LLMCouncil(providers=["openai", "broken"])
        folders, confidences = council.categorize(files)

    assert folders == ["Documents"]
    # Only one provider succeeded -> confidence should be 0.5 per spec.
    assert confidences == [0.5]


def test_council_all_providers_fail_returns_empty(monkeypatch):
    files = _files(1)

    clients = {
        "broken1": _FailingClient(),
        "broken2": _FailingClient(),
    }

    def fake_create(provider, api_key=None, model=None):
        return clients[provider]

    with patch("ai_rganize.council.council.create_ai_client", side_effect=fake_create):
        council = LLMCouncil(providers=["broken1", "broken2"])
        folders, confidences = council.categorize(files)

    assert folders == []
    assert confidences == []


def test_council_single_provider_confidence_is_half(monkeypatch):
    files = _files(3)
    clients = {"openai": _FakeClient(["A", "B", "C"])}

    def fake_create(provider, api_key=None, model=None):
        return clients[provider]

    with patch("ai_rganize.council.council.create_ai_client", side_effect=fake_create):
        council = LLMCouncil(providers=["openai"])
        folders, confidences = council.categorize(files)

    assert folders == ["A", "B", "C"]
    assert confidences == [0.5, 0.5, 0.5]


def test_council_mismatched_result_length_is_skipped(monkeypatch):
    files = _files(2)
    clients = {
        "good": _FakeClient(["A", "B"]),
        "bad_length": _FakeClient(["OnlyOne"]),
    }

    def fake_create(provider, api_key=None, model=None):
        return clients[provider]

    with patch("ai_rganize.council.council.create_ai_client", side_effect=fake_create):
        council = LLMCouncil(providers=["good", "bad_length"])
        folders, confidences = council.categorize(files)

    assert folders == ["A", "B"]
    assert confidences == [0.5, 0.5]
