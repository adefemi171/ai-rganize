"""Human-in-the-loop review of an AI organization plan before execution.

An organization plan (as produced by ``BaseOrganizer.create_organization_plan``)
maps folder names to lists of file dicts. ``ReviewPlan`` wraps that structure in
a list of :class:`ReviewItem` records that a user can approve, reject, or edit
before anything is actually moved on disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIDENCE = 0.7
COLLISION_CONFIDENCE = 0.5

Status = str  # one of: pending, approved, rejected, edited


@dataclass
class ReviewItem:
    """A single proposed file move awaiting human review."""

    source: Path
    destination: Path
    folder: str
    confidence: float = DEFAULT_CONFIDENCE
    reason: str = ""
    collisions: bool = False
    status: Status = "pending"

    def approve(self) -> None:
        self.status = "approved"

    def reject(self) -> None:
        self.status = "rejected"

    def set_destination(self, new_destination: Path) -> None:
        self.destination = Path(new_destination)
        self.status = "edited"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": str(self.source),
            "destination": str(self.destination),
            "folder": self.folder,
            "confidence": self.confidence,
            "reason": self.reason,
            "collisions": self.collisions,
            "status": self.status,
        }


def _compute_confidence(file_info: dict[str, Any], collision: bool) -> float:
    confidence = file_info.get("confidence")
    if confidence is not None:
        return float(confidence)
    return COLLISION_CONFIDENCE if collision else DEFAULT_CONFIDENCE


def _compute_reason(file_info: dict[str, Any], folder: str) -> str:
    reason = file_info.get("reason")
    if reason:
        return str(reason)
    return f"Categorized into '{folder}' based on filename/content analysis."


class ReviewPlan:
    """A reviewable collection of proposed file moves."""

    def __init__(self, items: list[ReviewItem]):
        self.items: list[ReviewItem] = items

    @classmethod
    def from_organization_plan(cls, plan: dict[str, Any], target_dir: Path) -> "ReviewPlan":
        """Build a ``ReviewPlan`` from an organizer plan dict + target directory.

        The plan dict is expected to map folder names to lists of file info
        dicts (each containing at least ``path``/``name``), with an optional
        ``summary`` key that is ignored here.
        """
        target_dir = Path(target_dir)
        items: list[ReviewItem] = []

        for folder_name, files in plan.items():
            if folder_name == "summary" or not files:
                continue

            for file_info in files:
                source = Path(file_info["path"])
                dest_folder = target_dir / folder_name
                destination = dest_folder / source.name
                collision = destination.exists() and destination.resolve() != source.resolve()

                items.append(
                    ReviewItem(
                        source=source,
                        destination=destination,
                        folder=folder_name,
                        confidence=_compute_confidence(file_info, collision),
                        reason=_compute_reason(file_info, folder_name),
                        collisions=collision,
                    )
                )

        return cls(items)

    def approve_all(self) -> None:
        for item in self.items:
            if item.status == "pending":
                item.approve()

    def reject(self, source: Path | str) -> bool:
        """Reject the item whose source matches *source*. Returns True if found."""
        source_str = str(source)
        for item in self.items:
            if str(item.source) == source_str:
                item.reject()
                return True
        return False

    def approve(self, source: Path | str) -> bool:
        source_str = str(source)
        for item in self.items:
            if str(item.source) == source_str:
                item.approve()
                return True
        return False

    def edit_destination(self, source: Path | str, new_destination: Path | str) -> bool:
        """Override the destination for the item matching *source*."""
        source_str = str(source)
        for item in self.items:
            if str(item.source) == source_str:
                item.set_destination(Path(new_destination))
                return True
        return False

    def pending(self) -> list[ReviewItem]:
        return [i for i in self.items if i.status == "pending"]

    def approved(self) -> list[ReviewItem]:
        return [i for i in self.items if i.status in ("approved", "edited")]

    def rejected(self) -> list[ReviewItem]:
        return [i for i in self.items if i.status == "rejected"]

    def low_confidence(self, threshold: float = 0.6) -> list[ReviewItem]:
        return [i for i in self.items if i.confidence < threshold]

    def to_execution_plan(self) -> dict[str, list[dict[str, Any]]]:
        """Build an organizer-compatible plan dict containing only approved items."""
        plan: dict[str, list[dict[str, Any]]] = {}
        for item in self.approved():
            plan.setdefault(item.folder, []).append(
                {
                    "path": item.source,
                    "name": item.source.name,
                    "size": item.source.stat().st_size if item.source.exists() else 0,
                    "destination_override": item.destination,
                }
            )
        return plan

    def to_dict(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.items]

    def display(self) -> None:
        """Render the plan as a rich table (falls back to plain text)."""
        try:
            from rich.console import Console
            from rich.table import Table
        except ImportError:
            markers = {"pending": "?", "approved": "✅", "rejected": "❌", "edited": "✏️"}
            for item in self.items:
                marker = markers.get(item.status, "?")
                print(
                    f"{marker} [{item.confidence:.2f}] {item.source.name} "
                    f"-> {item.folder}/ ({item.reason})"
                )
            return

        table = Table(title="AI-rganize Review Plan")
        table.add_column("Status")
        table.add_column("File")
        table.add_column("Folder")
        table.add_column("Confidence", justify="right")
        table.add_column("Collision", justify="center")
        table.add_column("Reason")

        status_styles = {
            "pending": "[yellow]pending[/yellow]",
            "approved": "[green]approved[/green]",
            "rejected": "[red]rejected[/red]",
            "edited": "[cyan]edited[/cyan]",
        }

        for item in self.items:
            table.add_row(
                status_styles.get(item.status, item.status),
                item.source.name,
                item.folder,
                f"{item.confidence:.2f}",
                "⚠️" if item.collisions else "",
                item.reason,
            )

        Console().print(table)
