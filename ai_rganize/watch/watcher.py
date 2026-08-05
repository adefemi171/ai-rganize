"""Continuous folder watching with debounced batch callbacks.

Uses ``watchdog`` when it's installed for efficient OS-level filesystem
events, and transparently falls back to a simple polling implementation
(comparing directory snapshots on an interval) when it isn't. Either way the
public API (`OrganizationWatcher`) behaves identically.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from datetime import time as dt_time
from pathlib import Path
from typing import Callable, Optional

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    HAS_WATCHDOG = True
except ImportError:  # pragma: no cover - exercised when watchdog isn't installed
    Observer = None
    FileSystemEventHandler = object
    HAS_WATCHDOG = False

BatchCallback = Callable[[list[Path]], None]

DEFAULT_DEBOUNCE_SECONDS = 30
DEFAULT_POLL_INTERVAL_SECONDS = 5


@dataclass
class QuietHours:
    """A daily quiet-hours window, e.g. 22:00 -> 07:00 (wraps past midnight)."""

    start: dt_time
    end: dt_time

    @classmethod
    def parse(cls, spec: str) -> "QuietHours":
        """Parse a ``HH:MM-HH:MM`` string into a QuietHours instance."""
        start_str, end_str = spec.split("-")
        return cls(start=_parse_hhmm(start_str.strip()), end=_parse_hhmm(end_str.strip()))

    def contains(self, when: Optional[datetime] = None) -> bool:
        now = (when or datetime.now()).time()
        if self.start <= self.end:
            return self.start <= now < self.end
        # Wraps past midnight, e.g. 22:00-07:00
        return now >= self.start or now < self.end


def _parse_hhmm(text: str) -> dt_time:
    hour, minute = (int(part) for part in text.split(":"))
    return dt_time(hour=hour, minute=minute)


class _DebouncedBatcher:
    """Thread-safe accumulator that flushes collected paths after a quiet period."""

    def __init__(
        self,
        debounce_seconds: float,
        on_flush: Callable[[list[Path]], None],
        is_quiet: Callable[[], bool],
    ):
        self.debounce_seconds = debounce_seconds
        self.on_flush = on_flush
        self.is_quiet = is_quiet
        self._lock = threading.Lock()
        self._pending: dict[Path, None] = {}
        self._timer: Optional[threading.Timer] = None

    def add(self, path: Path) -> None:
        with self._lock:
            self._pending[path] = None
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.debounce_seconds, self._flush)
            self._timer.daemon = True
            self._timer.start()

    def _flush(self) -> None:
        with self._lock:
            paths = list(self._pending.keys())
            self._pending.clear()
            self._timer = None
        if not paths:
            return
        if self.is_quiet():
            # Re-queue for later instead of dropping the batch entirely.
            with self._lock:
                for p in paths:
                    self._pending[p] = None
                self._timer = threading.Timer(self.debounce_seconds, self._flush)
                self._timer.daemon = True
                self._timer.start()
            return
        self.on_flush(paths)

    def cancel(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None


if HAS_WATCHDOG:

    class _WatchdogHandler(FileSystemEventHandler):
        def __init__(self, batcher: _DebouncedBatcher):
            super().__init__()
            self.batcher = batcher

        def on_created(self, event):
            if not event.is_directory:
                self.batcher.add(Path(event.src_path))

        def on_modified(self, event):
            if not event.is_directory:
                self.batcher.add(Path(event.src_path))


class OrganizationWatcher:
    """Watches one or more directories and invokes a callback with debounced
    batches of created/modified files.

    Parameters
    ----------
    paths:
        Directories to watch (non-recursive contents change detection).
    on_batch_callback:
        Called with a ``list[Path]`` of files that changed once the debounce
        window has elapsed with no further activity.
    quiet_hours:
        Optional ``"HH:MM-HH:MM"`` string; batches are held (not dropped)
        while the current time falls in this window.
    debounce_seconds:
        How long to wait after the last observed change before flushing.
    poll_interval_seconds:
        Only used in the polling fallback (no watchdog installed).
    """

    def __init__(
        self,
        paths: list[Path | str],
        on_batch_callback: BatchCallback,
        quiet_hours: Optional[str] = None,
        debounce_seconds: float = DEFAULT_DEBOUNCE_SECONDS,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    ):
        self.paths = [Path(p) for p in paths]
        self.on_batch_callback = on_batch_callback
        self.quiet_hours = QuietHours.parse(quiet_hours) if quiet_hours else None
        self.debounce_seconds = debounce_seconds
        self.poll_interval_seconds = poll_interval_seconds

        self._batcher = _DebouncedBatcher(debounce_seconds, self._on_flush, self._in_quiet_hours)
        self._observer = None
        self._poll_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._snapshot: dict[Path, float] = {}

    def _in_quiet_hours(self) -> bool:
        return self.quiet_hours is not None and self.quiet_hours.contains()

    def _on_flush(self, paths: list[Path]) -> None:
        self.on_batch_callback(paths)

    def using_watchdog(self) -> bool:
        return HAS_WATCHDOG

    def start(self) -> None:
        """Start watching in the background (non-blocking)."""
        if HAS_WATCHDOG:
            self._start_watchdog()
        else:
            self._start_polling()

    def _start_watchdog(self) -> None:
        self._observer = Observer()
        handler = _WatchdogHandler(self._batcher)
        for path in self.paths:
            if path.exists():
                self._observer.schedule(handler, str(path), recursive=False)
        self._observer.start()

    def _snapshot_dir(self, path: Path) -> dict[Path, float]:
        snapshot = {}
        try:
            for entry in path.iterdir():
                if entry.is_file():
                    try:
                        snapshot[entry] = entry.stat().st_mtime
                    except OSError:
                        pass
        except OSError:
            pass
        return snapshot

    def _poll_loop(self) -> None:
        for path in self.paths:
            self._snapshot.update(self._snapshot_dir(path))

        while not self._stop_event.is_set():
            self._stop_event.wait(self.poll_interval_seconds)
            if self._stop_event.is_set():
                break
            current: dict[Path, float] = {}
            for path in self.paths:
                current.update(self._snapshot_dir(path))

            for file_path, mtime in current.items():
                previous = self._snapshot.get(file_path)
                if previous is None or mtime > previous:
                    self._batcher.add(file_path)

            self._snapshot = current

    def _start_polling(self) -> None:
        self._stop_event.clear()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

    def stop(self) -> None:
        self._batcher.cancel()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
        if self._poll_thread is not None:
            self._stop_event.set()
            self._poll_thread.join(timeout=self.poll_interval_seconds + 2)
            self._poll_thread = None

    def run_forever(self) -> None:
        """Start watching and block the calling thread until interrupted."""
        self.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
