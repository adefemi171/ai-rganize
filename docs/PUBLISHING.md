# Publishing AI-rganize (PyPI · GitHub · Homebrew)

## 1. PyPI (`pip` / `uv tool install`)

### One-time PyPI Trusted Publishing setup

1. Create/claim the project on [PyPI](https://pypi.org) (name: `ai_rganize`).
2. Under **Publishing** → **Add a new pending publisher**:
   - Owner: `adefemi171`
   - Repository: `airganizer` (match the GitHub repo name)
   - Workflow name: `publish-pypi.yml`
   - Environment name: `pypi`
3. In GitHub → **Settings → Environments**, create environment `pypi` (optional protection rules).

### Release flow

```bash
# Ensure version in pyproject.toml / ai_rganize/__init__.py matches the tag
git tag -a v1.0.0 -m "v1.0.0"
git push origin v1.0.0
```

This triggers:

1. **Release** workflow → GitHub Release + wheel/sdist artifacts  
2. **Publish to PyPI** workflow (on `release: published`) → uploads to PyPI  

Users can then:

```bash
uv tool install ai_rganize
pip install ai_rganize
```

Dry-run a build without uploading: Actions → **Publish to PyPI** → Run workflow → `dry_run: true`.

---

## 2. GitHub / `uv tool` (works before PyPI)

```bash
# Latest main
uv tool install "git+https://github.com/adefemi171/airganizer.git"

# Pin to a tag
uv tool install "ai_rganize @ git+https://github.com/adefemi171/airganizer.git@v1.0.0"

# Upgrade later
uv tool upgrade ai_rganize
```

`pipx` equivalent:

```bash
pipx install "git+https://github.com/adefemi171/airganizer.git@v1.0.0"
```

---

## 3. Homebrew (macOS)

### One-time tap repo

Create a public repo: `https://github.com/adefemi171/homebrew-airganize`

```text
homebrew-airganize/
  Formula/
    ai-rganize.rb    # copy from this repo's homebrew/ai-rganize.rb
  README.md
```

### After each tagged release

```bash
./scripts/update_homebrew_sha.sh v1.0.0
# Copy homebrew/ai-rganize.rb → homebrew-airganize/Formula/ai-rganize.rb
# Commit + push the tap
```

### User install

```bash
brew tap adefemi171/airganize
brew install ai-rganize
brew install ffmpeg   # if not pulled as recommended dependency
```

Local formula test (from this repo, before the tap exists):

```bash
# After a real tag exists and sha256 is updated:
brew install --formula ./homebrew/ai-rganize.rb
```

---

## Checklist for v1.0.0

- [ ] Version is `1.0.0` in `pyproject.toml` and `ai_rganize/__init__.py`
- [ ] CI green on `main`
- [ ] PyPI trusted publisher configured
- [ ] Tag `v1.0.0` pushed
- [ ] GitHub Release published; PyPI publish job green
- [ ] `uv tool install ai_rganize` works
- [ ] Homebrew sha updated + tap pushed
- [ ] `brew install ai-rganize` works on a clean Mac
