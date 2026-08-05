"""Tests for ai_rganize.review.plan (ReviewItem / ReviewPlan)."""

from __future__ import annotations

from pathlib import Path

from ai_rganize.review.plan import (
    COLLISION_CONFIDENCE,
    DEFAULT_CONFIDENCE,
    ReviewItem,
    ReviewPlan,
)


def _make_plan_dict(source_dir: Path):
    file_a = source_dir / "invoice.pdf"
    file_b = source_dir / "photo.png"
    file_a.write_text("pdf-bytes")
    file_b.write_text("png-bytes")

    return {
        "Finance": [{"path": file_a, "name": "invoice.pdf", "size": 9}],
        "Photos": [{"path": file_b, "name": "photo.png", "size": 9}],
        "summary": {"total_files": 2},
    }


def test_from_organization_plan_builds_items(tmp_path):
    plan_dict = _make_plan_dict(tmp_path)
    target_dir = tmp_path / "Organized"

    plan = ReviewPlan.from_organization_plan(plan_dict, target_dir)

    assert len(plan.items) == 2
    folders = {item.folder for item in plan.items}
    assert folders == {"Finance", "Photos"}
    assert all(item.status == "pending" for item in plan.items)


def test_default_confidence_used_when_missing(tmp_path):
    plan_dict = _make_plan_dict(tmp_path)
    plan = ReviewPlan.from_organization_plan(plan_dict, tmp_path / "Organized")

    for item in plan.items:
        assert item.confidence == DEFAULT_CONFIDENCE


def test_collision_lowers_confidence(tmp_path):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    target_dir = tmp_path / "Organized"
    (target_dir / "Finance").mkdir(parents=True)

    source_file = source_dir / "invoice.pdf"
    source_file.write_text("data")
    # Pre-existing file at the destination triggers a collision.
    (target_dir / "Finance" / "invoice.pdf").write_text("existing")

    plan_dict = {"Finance": [{"path": source_file, "name": "invoice.pdf", "size": 4}]}
    plan = ReviewPlan.from_organization_plan(plan_dict, target_dir)

    item = plan.items[0]
    assert item.collisions is True
    assert item.confidence == COLLISION_CONFIDENCE


def test_explicit_confidence_is_respected(tmp_path):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    source_file = source_dir / "a.txt"
    source_file.write_text("x")

    plan_dict = {"Docs": [{"path": source_file, "name": "a.txt", "size": 1, "confidence": 0.95}]}
    plan = ReviewPlan.from_organization_plan(plan_dict, tmp_path / "Organized")

    assert plan.items[0].confidence == 0.95


def test_approve_all(tmp_path):
    plan_dict = _make_plan_dict(tmp_path)
    plan = ReviewPlan.from_organization_plan(plan_dict, tmp_path / "Organized")

    plan.approve_all()
    assert all(item.status == "approved" for item in plan.items)
    assert len(plan.approved()) == 2


def test_reject_specific_item(tmp_path):
    plan_dict = _make_plan_dict(tmp_path)
    plan = ReviewPlan.from_organization_plan(plan_dict, tmp_path / "Organized")

    target_source = plan.items[0].source
    assert plan.reject(target_source) is True
    assert plan.items[0].status == "rejected"
    assert len(plan.rejected()) == 1


def test_reject_unknown_source_returns_false(tmp_path):
    plan_dict = _make_plan_dict(tmp_path)
    plan = ReviewPlan.from_organization_plan(plan_dict, tmp_path / "Organized")

    assert plan.reject("/nonexistent/path.txt") is False


def test_edit_destination(tmp_path):
    plan_dict = _make_plan_dict(tmp_path)
    plan = ReviewPlan.from_organization_plan(plan_dict, tmp_path / "Organized")

    target_source = plan.items[0].source
    new_dest = tmp_path / "CustomFolder" / "renamed.pdf"
    assert plan.edit_destination(target_source, new_dest) is True

    edited_item = plan.items[0]
    assert edited_item.destination == new_dest
    assert edited_item.status == "edited"


def test_to_execution_plan_only_includes_approved(tmp_path):
    plan_dict = _make_plan_dict(tmp_path)
    plan = ReviewPlan.from_organization_plan(plan_dict, tmp_path / "Organized")

    # Approve only one item, reject the other.
    plan.approve(plan.items[0].source)
    plan.reject(plan.items[1].source)

    execution_plan = plan.to_execution_plan()
    total_files = sum(len(files) for files in execution_plan.values())
    assert total_files == 1
    assert plan.items[0].folder in execution_plan


def test_low_confidence_filter(tmp_path):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    f1 = source_dir / "a.txt"
    f2 = source_dir / "b.txt"
    f1.write_text("1")
    f2.write_text("2")

    plan_dict = {
        "Docs": [
            {"path": f1, "name": "a.txt", "size": 1, "confidence": 0.3},
            {"path": f2, "name": "b.txt", "size": 1, "confidence": 0.9},
        ]
    }
    plan = ReviewPlan.from_organization_plan(plan_dict, tmp_path / "Organized")

    low = plan.low_confidence(threshold=0.6)
    assert len(low) == 1
    assert low[0].source == f1


def test_review_item_to_dict_serializes_paths_as_strings():
    item = ReviewItem(
        source=Path("/a/b.txt"),
        destination=Path("/c/b.txt"),
        folder="Docs",
    )
    data = item.to_dict()
    assert data["source"] == "/a/b.txt"
    assert data["destination"] == "/c/b.txt"
    assert data["status"] == "pending"


def test_display_does_not_raise(tmp_path):
    plan_dict = _make_plan_dict(tmp_path)
    plan = ReviewPlan.from_organization_plan(plan_dict, tmp_path / "Organized")
    plan.display()  # Should not raise regardless of whether rich is installed.
