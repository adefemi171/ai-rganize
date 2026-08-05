"""Multi-provider "council" categorization.

Rather than trusting a single LLM provider's categorization, ``LLMCouncil``
asks several providers to categorize the same batch of files and aggregates
their answers by majority vote per file, producing a confidence score based
on how much the providers agreed.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Optional

from ai_rganize.ai_client import create_ai_client


class LLMCouncil:
    """Query multiple LLM providers and aggregate their categorizations."""

    def __init__(
        self,
        providers: list[str],
        api_key: Optional[str] = None,
        models: Optional[dict[str, str]] = None,
        api_keys: Optional[dict[str, str]] = None,
    ):
        if not providers:
            raise ValueError("LLMCouncil requires at least one provider")
        self.providers = list(providers)
        self.api_key = api_key
        self.models = models or {}
        self.api_keys = api_keys or {}

    def _client_for(self, provider: str):
        key = self.api_keys.get(provider, self.api_key)
        model = self.models.get(provider)
        return create_ai_client(provider, key, model)

    def categorize(
        self,
        file_batch: list[dict[str, Any]],
        verbose: bool = False,
        max_folders: Optional[int] = None,
    ) -> tuple[list[str], list[float]]:
        """Ask every configured provider to categorize *file_batch*.

        Returns a tuple of ``(folder_names, confidences)`` aligned by index
        with *file_batch*. ``confidences[i]`` is the fraction of successful
        providers that agreed with the winning folder name for file ``i``.
        Providers that fail to initialize or error out are skipped entirely;
        if every provider fails, both lists are empty.
        """
        n = len(file_batch)
        per_provider_votes: list[list[str]] = []

        for provider in self.providers:
            try:
                client = self._client_for(provider)
                folders = client.categorize_files(
                    file_batch, verbose=verbose, max_folders=max_folders
                )
                if len(folders) != n:
                    if verbose:
                        print(
                            f"    ⚠️  Council: {provider} returned {len(folders)} results "
                            f"for {n} files, skipping"
                        )
                    continue
                per_provider_votes.append(folders)
            except Exception as exc:  # noqa: BLE001 - provider failures are expected/handled
                if verbose:
                    print(f"    ⚠️  Council: provider '{provider}' failed: {exc}")
                continue

        if not per_provider_votes:
            return [], []

        folder_names: list[str] = []
        confidences: list[float] = []
        num_successful = len(per_provider_votes)

        for i in range(n):
            votes = [provider_result[i] for provider_result in per_provider_votes]
            counts = Counter(votes)
            winner, winner_count = counts.most_common(1)[0]
            folder_names.append(winner)

            if num_successful == 1:
                confidences.append(0.5)
            else:
                confidences.append(winner_count / num_successful)

        return folder_names, confidences

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"LLMCouncil(providers={self.providers!r})"
