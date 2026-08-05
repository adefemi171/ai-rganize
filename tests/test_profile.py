"""Tests for ai_rganize.config.profile."""

from __future__ import annotations

import pytest

from ai_rganize.config.profile import (
    Profile,
    default_profile,
    delete_profile,
    find_project_profile,
    list_profiles,
    load_profile,
    load_project_profile,
    resolve_effective_profile,
    resolve_profile_path,
    save_profile,
    save_project_profile,
)


def test_save_and_load_profile_roundtrip(isolated_home):
    profile = Profile(
        name="work",
        roots=["/Users/me/Downloads"],
        destination="/Users/me/Organized",
        provider="claude",
        model="claude-sonnet-4",
        max_cost=2.5,
        max_folders=8,
        exclusions=["*.tmp"],
        schedule="0 9 * * MON",
        enable_council=True,
        auto_unpack_archives=True,
        cloud_providers=["dropbox"],
    )
    save_profile(profile)

    loaded = load_profile("work")

    assert loaded == profile


def test_load_missing_profile_raises(isolated_home):
    with pytest.raises(FileNotFoundError):
        load_profile("does_not_exist")


def test_list_profiles_returns_sorted_names(isolated_home):
    save_profile(Profile(name="zeta"))
    save_profile(Profile(name="alpha"))

    assert list_profiles() == ["alpha", "zeta"]


def test_list_profiles_empty_when_none_saved(isolated_home):
    assert list_profiles() == []


def test_default_profile_created_on_first_access(isolated_home):
    profile = default_profile()
    assert profile.name == "default"
    assert "default" in list_profiles()

    # Second call should load the same profile, not create a duplicate.
    profile2 = default_profile()
    assert profile2 == profile


def test_resolve_profile_path_prefers_yaml_when_available(isolated_home):
    path = resolve_profile_path("brand_new")
    assert path.suffix in (".yaml", ".json")


def test_delete_profile_removes_file(isolated_home):
    save_profile(Profile(name="temp"))
    assert "temp" in list_profiles()

    removed = delete_profile("temp")
    assert removed is True
    assert "temp" not in list_profiles()


def test_delete_missing_profile_returns_false(isolated_home):
    assert delete_profile("never_existed") is False


def test_project_local_profile_roundtrip(tmp_path):
    project_dir = tmp_path / "my_project"
    project_dir.mkdir()
    profile = Profile(name="project", roots=[str(project_dir)])

    saved_path = save_project_profile(profile, directory=project_dir)
    assert saved_path.exists()

    loaded = load_project_profile(project_dir)
    assert loaded is not None
    assert loaded.name == "project"


def test_find_project_profile_searches_ancestors(tmp_path):
    project_dir = tmp_path / "my_project"
    nested_dir = project_dir / "sub" / "deep"
    nested_dir.mkdir(parents=True)

    save_project_profile(Profile(name="proj"), directory=project_dir)

    found = find_project_profile(nested_dir)
    assert found is not None
    assert found.parent == project_dir


def test_find_project_profile_returns_none_when_absent(tmp_path):
    isolated_dir = tmp_path / "no_profile_here"
    isolated_dir.mkdir()
    assert find_project_profile(isolated_dir) is None


def test_resolve_effective_profile_prefers_explicit_name(isolated_home):
    save_profile(Profile(name="explicit"))
    resolved = resolve_effective_profile(name="explicit")
    assert resolved.name == "explicit"


def test_resolve_effective_profile_falls_back_to_project_local(isolated_home, tmp_path):
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    save_project_profile(Profile(name="local_project"), directory=project_dir)

    resolved = resolve_effective_profile(start_dir=project_dir)
    assert resolved.name == "local_project"


def test_resolve_effective_profile_falls_back_to_default(isolated_home, tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    resolved = resolve_effective_profile(start_dir=empty_dir)
    assert resolved.name == "default"
