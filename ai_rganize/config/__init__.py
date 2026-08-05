"""Profile and exclusion configuration for AI-rganize."""

from .exclusions import (
    DEFAULT_PROTECTED_PATTERNS,
    ExclusionMatcher,
    build_matcher_for_directory,
    find_ignore_file,
    load_ignore_file,
)
from .profile import (
    DEFAULT_PROFILE_NAME,
    PROFILE_HOME,
    PROFILES_DIR,
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

__all__ = [
    "Profile",
    "DEFAULT_PROFILE_NAME",
    "PROFILE_HOME",
    "PROFILES_DIR",
    "default_profile",
    "delete_profile",
    "find_project_profile",
    "load_profile",
    "load_project_profile",
    "list_profiles",
    "resolve_effective_profile",
    "resolve_profile_path",
    "save_profile",
    "save_project_profile",
    "DEFAULT_PROTECTED_PATTERNS",
    "ExclusionMatcher",
    "build_matcher_for_directory",
    "find_ignore_file",
    "load_ignore_file",
]
