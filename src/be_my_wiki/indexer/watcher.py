"""Filesystem watcher that drives incremental re-indexing.

watchdog runs a native filesystem-event observer in a background thread.
We translate those events into ``UPSERT`` / ``DELETE`` records and let a
worker thread debounce and dispatch them to the Indexer. Debouncing
matters because editors typically emit multiple ``modified`` events for
a single save (e.g. write-then-rename), and rapid edits would otherwise
re-embed the same file repeatedly.

For tests the public ``enqueue`` and ``_flush_pending`` methods can be
called directly, bypassing the watchdog observer and worker thread for
deterministic behaviour. ``run_forever`` is the production entry point.
"""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver

if TYPE_CHECKING:
    from .pipeline import Indexer

logger = logging.getLogger(__name__)


class EventKind(Enum):
    UPSERT = "upsert"
    DELETE = "delete"


class VaultWatcher:
    def __init__(
        self,
        *,
        indexer: "Indexer",
        debounce_seconds: float = 2.0,
        polling: bool = False,
    ) -> None:
        self.indexer = indexer
        self.debounce_seconds = debounce_seconds
        self._polling = polling
        self._pending: dict[Path, tuple[EventKind, float]] = {}
        self._lock = threading.Lock()
        self._observer = None
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        observer_cls = PollingObserver if self._polling else Observer
        self._observer = observer_cls()
        self._observer.schedule(
            _WatchdogHandler(self),
            path=str(self.indexer.vault_root),
            recursive=True,
        )
        self._observer.start()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._stop.set()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
        if self._worker is not None:
            self._worker.join(timeout=5)
        self._flush_pending(force=True)

    def run_forever(self) -> None:
        self.start()
        try:
            while not self._stop.wait(1.0):
                pass
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def enqueue(self, kind: EventKind, abs_path: Path) -> None:
        """Record an event for deferred processing.

        Public for testing — production code paths reach this through the
        watchdog handler. Filters non-md files and ignored directories.
        """
        if not self._is_relevant(abs_path):
            return
        with self._lock:
            self._pending[abs_path] = (kind, time.monotonic())

    def _worker_loop(self) -> None:
        interval = max(self.debounce_seconds / 2, 0.05)
        while not self._stop.is_set():
            self._flush_pending()
            self._stop.wait(timeout=interval)

    def _flush_pending(self, *, force: bool = False) -> None:
        now = time.monotonic()
        ready: list[tuple[Path, EventKind]] = []
        with self._lock:
            for path, (kind, ts) in list(self._pending.items()):
                if force or (now - ts) >= self.debounce_seconds:
                    ready.append((path, kind))
                    self._pending.pop(path, None)
        for path, kind in ready:
            try:
                self._process(path, kind)
            except Exception as exc:
                logger.warning(
                    "Failed to process %s (%s): %s", path, kind.value, exc
                )

    def _process(self, abs_path: Path, kind: EventKind) -> None:
        if kind == EventKind.DELETE or not abs_path.exists():
            self.indexer.delete_note(abs_path)
            logger.info("deleted %s", abs_path)
            return
        result = self.indexer.index_note(abs_path)
        logger.info(
            "indexed %s (added=%d updated=%d skipped=%d deleted=%d)",
            result.note_path,
            result.added,
            result.updated,
            result.skipped,
            result.deleted,
        )

    def _is_relevant(self, abs_path: Path) -> bool:
        if abs_path.suffix.lower() != ".md":
            return False
        try:
            rel = abs_path.resolve().relative_to(
                self.indexer.vault_root.resolve()
            )
        except ValueError:
            return False
        if any(part in self.indexer.ignore_dirs for part in rel.parts):
            return False
        return True


class _WatchdogHandler(FileSystemEventHandler):
    def __init__(self, watcher: VaultWatcher) -> None:
        self.watcher = watcher

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self.watcher.enqueue(EventKind.UPSERT, Path(event.src_path))

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self.watcher.enqueue(EventKind.UPSERT, Path(event.src_path))

    def on_deleted(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self.watcher.enqueue(EventKind.DELETE, Path(event.src_path))

    def on_moved(self, event) -> None:
        if event.is_directory:
            return
        self.watcher.enqueue(EventKind.DELETE, Path(event.src_path))
        self.watcher.enqueue(EventKind.UPSERT, Path(event.dest_path))
