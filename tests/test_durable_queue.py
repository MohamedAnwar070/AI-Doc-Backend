from __future__ import annotations

import queue
import threading
import time
import concurrent.futures
from pathlib import Path

from core.job_manager import JobManager
from storage.file_handler import FileHandler


class FakeStorage:
    def __init__(self, job_ids: list[str] | None = None) -> None:
        self.job_ids = job_ids or []

    def list_jobs(self) -> list[str]:
        return list(self.job_ids)

    def job_exists(self, job_id: str) -> bool:
        return job_id in self.job_ids


class FakeProgressTracker:
    def __init__(self, statuses: dict[str, dict]) -> None:
        self.statuses = statuses
        self.failures: list[tuple[str, str]] = []
        self.pending: list[str] = []
        self.indexing_failures: list[tuple[str, str, int]] = []

    def get_status(self, job_id: str) -> dict:
        return self.statuses[job_id]

    def mark_failed(self, job_id: str, error: str) -> None:
        self.failures.append((job_id, error))

    def mark_pending(self, job_id: str) -> None:
        self.pending.append(job_id)
        self.statuses[job_id]["status"] = "pending"

    def mark_indexing_failed(self, job_id: str, error: str, version: int) -> None:
        self.indexing_failures.append((job_id, error, version))
        self.statuses[job_id]["search_index_status"] = "failed"


class FakeResumeHandler:
    def __init__(self) -> None:
        self.marked: list[str] = []

    def mark_for_resume(self, job_id: str) -> None:
        self.marked.append(job_id)


def make_manager(storage, tracker) -> JobManager:
    manager = JobManager.__new__(JobManager)
    manager.storage = storage
    manager.progress_tracker = tracker
    manager.resume_handler = FakeResumeHandler()
    manager.upload_dir = Path("uploads")
    manager.loop = None
    manager._processing_queue = queue.Queue()
    manager._queued_job_ids = set()
    manager._queue_lock = threading.Lock()
    manager._queue_started = False
    manager._indexing_futures = {}
    manager._indexing_lock = threading.Lock()
    manager.cleanup_stale_transfers = lambda max_age_seconds=0: 0  # type: ignore[method-assign]
    return manager


def test_source_files_are_isolated_by_job_and_atomically_replaced(tmp_path) -> None:
    storage = FileHandler(str(tmp_path / "outputs"))
    first = tmp_path / "first" / "same.pdf"
    second = tmp_path / "second" / "same.pdf"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"%PDF-first")
    second.write_bytes(b"%PDF-second")
    storage.create_job_directory("job-a")
    storage.create_job_directory("job-b")

    path_a = storage.save_source_file("job-a", first)
    path_b = storage.save_source_file("job-b", second)

    assert path_a.read_bytes() == b"%PDF-first"
    assert path_b.read_bytes() == b"%PDF-second"
    assert not list((tmp_path / "outputs").rglob("*.tmp"))


def test_page_retry_replaces_text_without_duplication(tmp_path) -> None:
    storage = FileHandler(str(tmp_path / "outputs"))
    storage.create_job_directory("job-a")

    storage.append_text("job-a", "old page one", 1)
    storage.append_text("job-a", "page two", 2)
    storage.append_text("job-a", "corrected page one", 1)

    result = storage.get_result_file_path("job-a").read_text(encoding="utf-8")
    assert result.count("--- Page 1 ---") == 1
    assert "old page one" not in result
    assert "corrected page one" in result
    assert result.index("--- Page 1 ---") < result.index("--- Page 2 ---")


def test_cleanup_only_removes_stale_unique_transfers(tmp_path) -> None:
    manager = make_manager(FakeStorage(), FakeProgressTracker({}))
    manager.upload_dir = tmp_path
    manager.cleanup_stale_transfers = JobManager.cleanup_stale_transfers.__get__(manager)
    partial = tmp_path / ".0123456789abcdef0123456789abcdef.part"
    finalized = tmp_path / "0123456789abcdef0123456789abcdef-document.pdf"
    legacy = tmp_path / "document.pdf"
    for path in (partial, finalized, legacy):
        path.write_bytes(b"data")

    removed = manager.cleanup_stale_transfers(max_age_seconds=-1)

    assert removed == 2
    assert not partial.exists()
    assert not finalized.exists()
    assert legacy.exists()


def test_startup_requeues_pending_and_running_but_not_failed_jobs() -> None:
    statuses = {
        "pending": {"status": "pending"},
        "running": {"status": "running"},
        "failed": {"status": "failed"},
        "done": {"status": "completed", "search_index_status": "indexed"},
    }
    manager = make_manager(FakeStorage(list(statuses)), FakeProgressTracker(statuses))

    recovered = manager.recover_interrupted_jobs()

    assert recovered == 2
    assert manager.progress_tracker.pending == ["pending", "running"]
    assert list(manager._processing_queue.queue) == ["pending", "running"]


def test_startup_does_not_launch_hidden_interrupted_indexing() -> None:
    statuses = {
        "pending-index": {"status": "completed", "search_index_status": "pending"},
        "failed-index": {"status": "completed", "search_index_status": "failed"},
        "active-index": {"status": "completed", "search_index_status": "indexing"},
    }
    manager = make_manager(FakeStorage(list(statuses)), FakeProgressTracker(statuses))
    scheduled: list[str] = []
    manager._schedule_index_recovery = scheduled.append  # type: ignore[method-assign]

    manager.recover_interrupted_jobs()

    assert scheduled == []
    assert statuses["pending-index"]["search_index_status"] == "pending"
    assert statuses["failed-index"]["search_index_status"] == "failed"
    assert statuses["active-index"]["search_index_status"] == "failed"


def test_cancel_completed_job_cancels_its_indexing_without_failing_ocr() -> None:
    statuses = {
        "job-a": {"status": "completed", "search_index_status": "indexing"},
    }
    tracker = FakeProgressTracker(statuses)
    manager = make_manager(FakeStorage(["job-a"]), tracker)
    future: concurrent.futures.Future = concurrent.futures.Future()
    manager._indexing_futures["job-a"] = future

    assert manager.cancel_job("job-a") is True
    assert future.cancelled()
    assert statuses["job-a"]["status"] == "completed"
    assert statuses["job-a"]["search_index_status"] == "failed"


def test_client_upload_id_returns_existing_job_for_safe_retry() -> None:
    statuses = {
        "job-a": {"job_id": "job-a", "status": "pending", "client_upload_id": "upload-123"},
        "job-b": {"job_id": "job-b", "status": "completed", "client_upload_id": "upload-456"},
    }
    manager = make_manager(FakeStorage(list(statuses)), FakeProgressTracker(statuses))

    assert manager.find_job_by_client_upload_id("upload-123") == statuses["job-a"]
    assert manager.find_job_by_client_upload_id("missing") is None


def test_queue_has_exactly_one_active_processor() -> None:
    job_ids = ["one", "two", "three"]
    manager = make_manager(
        FakeStorage(job_ids),
        FakeProgressTracker({job_id: {"status": "pending"} for job_id in job_ids}),
    )
    active = 0
    maximum_active = 0
    completed: list[str] = []
    state_lock = threading.Lock()

    def process(job_id: str) -> None:
        nonlocal active, maximum_active
        with state_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.02)
        with state_lock:
            completed.append(job_id)
            active -= 1

    manager._process_queued_job = process  # type: ignore[method-assign]
    manager.start_queue()
    for job_id in job_ids:
        manager.enqueue_job(job_id)
    manager._processing_queue.join()

    assert completed == job_ids
    assert maximum_active == 1


def test_source_extension_is_type_aware_for_legacy_statuses() -> None:
    assert JobManager._source_extension({"engine": "word"}) == "docx"
    assert JobManager._source_extension({"engine": "excel"}) == "xlsx"
    assert JobManager._source_extension({"engine": "google_docai"}) == "pdf"
    assert JobManager._source_extension({"engine": "word", "source_extension": "pdf"}) == "pdf"


def test_resolve_source_prefers_job_scoped_document(tmp_path) -> None:
    storage = FileHandler(str(tmp_path / "outputs"))
    storage.create_job_directory("word-job")
    scoped = storage.get_source_file_path("word-job", "docx")
    scoped.write_bytes(b"docx")
    tracker = FakeProgressTracker({
        "word-job": {"job_id": "word-job", "status": "pending", "engine": "word"}
    })
    manager = make_manager(storage, tracker)
    manager.upload_dir = tmp_path / "uploads"
    manager.upload_dir.mkdir()

    assert manager.resolve_source_path("word-job") == scoped
