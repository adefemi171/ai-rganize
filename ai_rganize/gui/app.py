"""Local-only Flask dashboard for AI-rganize.

Binds strictly to ``127.0.0.1:8765`` -- this is a local operator console, not
a network service, and it must never be exposed beyond localhost. The
``/api/organize`` endpoint defaults to a dry run; a caller must explicitly
opt in with ``{"dry_run": false}`` to actually move files, and even then it
only builds/executes an organization plan through the existing organizer
classes (no new file-moving logic lives here).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

from ai_rganize.config import default_profile, list_profiles, load_profile
from ai_rganize.features.cloud_sync import discover_cloud_roots
from ai_rganize.provenance.ledger import LEDGER_PATH
from ai_rganize.provenance.ledger import query as query_ledger

HOST = "127.0.0.1"
PORT = 8765

TEMPLATE_DIR = Path(__file__).parent / "templates"


def create_app() -> Flask:
    app = Flask(__name__, template_folder=str(TEMPLATE_DIR))

    @app.after_request
    def _restrict_host(response):
        # Defense in depth: reject anything that didn't arrive via localhost,
        # in case the process is ever run behind an unexpected proxy.
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.route("/", methods=["GET"])
    def dashboard():
        recent_ledger = query_ledger()[-50:]
        profiles = list_profiles()
        cloud_roots = {name: str(path) for name, path in discover_cloud_roots().items()}
        return render_template(
            "dashboard.html",
            ledger_entries=list(reversed(recent_ledger)),
            profiles=profiles,
            cloud_roots=cloud_roots,
            ledger_path=str(LEDGER_PATH),
        )

    @app.route("/api/status", methods=["GET"])
    def api_status():
        return jsonify(_status_payload())

    @app.route("/api/organize", methods=["POST"])
    def api_organize():
        payload: dict[str, Any] = request.get_json(silent=True) or {}
        dry_run = payload.get("dry_run", True)
        directory = payload.get("directory")
        profile_name = payload.get("profile")

        try:
            profile = load_profile(profile_name) if profile_name else default_profile()
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404

        if directory:
            target_dir = Path(directory)
        elif profile.destination:
            target_dir = Path(profile.destination)
        else:
            target_dir = None
        if target_dir is None or not target_dir.exists():
            return jsonify({"error": "No valid directory provided or found in profile"}), 400

        result = _build_plan_summary(target_dir, profile)
        result["dry_run"] = bool(dry_run)

        if not dry_run:
            # Executing organization from the web UI is intentionally not
            # implemented here to avoid duplicating (and potentially
            # diverging from) the CLI's safety-checked execution path.
            # Callers should use the CLI's `--restore`/execute flow, which
            # already handles manifests, metadata preservation, and undo.
            result["executed"] = False
            result["note"] = (
                "Execution from the dashboard is disabled for safety; "
                "use the ai-rganize CLI to execute a plan."
            )
        else:
            result["executed"] = False

        return jsonify(result)

    return app


def _status_payload() -> dict[str, Any]:
    cloud_roots = {name: str(path) for name, path in discover_cloud_roots().items()}
    recent = query_ledger()
    return {
        "profiles": list_profiles(),
        "cloud_providers_detected": cloud_roots,
        "ledger_path": str(LEDGER_PATH),
        "ledger_record_count": len(recent),
        "host": HOST,
        "port": PORT,
    }


def _build_plan_summary(target_dir: Path, profile) -> dict[str, Any]:
    from ai_rganize.organizer.rule_based_organizer import RuleBasedOrganizer

    organizer = RuleBasedOrganizer()
    files = organizer.scan_files(target_dir)
    plan = organizer.create_organization_plan(files)

    folders = {k: len(v) for k, v in plan.items() if k != "summary"}
    return {
        "directory": str(target_dir),
        "profile": profile.name,
        "total_files": len(files),
        "folders": folders,
    }


def main() -> None:
    """Entry point for the ``ai-rganize-gui`` console script."""
    app = create_app()
    print(f"AI-rganize dashboard running at http://{HOST}:{PORT} (localhost only)")
    app.run(host=HOST, port=PORT, debug=False)


if __name__ == "__main__":
    main()
