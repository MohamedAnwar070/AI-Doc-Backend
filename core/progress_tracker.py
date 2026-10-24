from typing import Dict, Any, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Track progress of OCR jobs"""

    def __init__(self, file_handler: Any):
        self.file_handler = file_handler

    def initialize_job(
        self,
        job_id: str,
        total_pages: int,
        pdf_filename: str,
        engine: str,
        language: str = "en",
        doc_type: str = "auto",
        category: str = "",
        categories: list[str] | None = None,
        tags: list[str] | None = None,
        sheet_name: str = "",
        source_total_pages: int | None = None,
        source_preview_error: str = "",
        source_extension: str = "pdf",
        client_upload_id: str = "",
    ) -> None:
        """
        Initialize job tracking

        Args:
            job_id: Job identifier
            total_pages: Total number of pages
            pdf_filename: Original PDF filename
            engine: OCR engine being used
            language: OCR language
        """
        status_data = {
            "status": "pending",
            "job_id": job_id,
            "pdf_filename": pdf_filename,
            "total_pages": total_pages,
            "source_total_pages": source_total_pages or total_pages,
            "source_preview_error": source_preview_error,
            "source_extension": source_extension,
            "client_upload_id": client_upload_id,
            "processed_pages": [],
            "failed_pages": [],
            "percentage": 0.0,
            "engine": engine,
            "language": language,
            "doc_type": doc_type,
            "category": category,
            "categories": categories or ([category] if category else []),
            "tags": tags or [],
            "sheet_name": sheet_name,
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "error": None,
            "search_indexed": False,
            "search_index_status": "pending",
            "search_index_started_at": None,
            "search_indexed_at": None,
            "search_index_error": None,
            "search_index_version": None,
            "search_chunk_count": 0,
            "search_expected_chunk_count": 0,
            "search_saved_chunk_count": 0,
            "search_embedded_chunk_count": 0,
            "search_embedding_failed_count": 0,
            "search_index_health": "pending",
        }

        self.file_handler.save_status(job_id, status_data)
        self._sync_to_db(job_id)
        logger.info(f"Initialized job {job_id} with {total_pages} pages")

    def start_job(self, job_id: str) -> None:
        """Mark job as started"""
        status = self.file_handler.load_status(job_id)
        if status:
            status["status"] = "running"
            status["started_at"] = datetime.now().isoformat()
            status["completed_at"] = None
            status["error"] = None
            status["failed_pages"] = []
            self.file_handler.save_status(job_id, status)
            self._sync_to_db(job_id)

    def update_progress(self, job_id: str, page_num: int, success: bool = True) -> None:
        """
        Update job progress after processing a page

        Args:
            job_id: Job identifier
            page_num: Page number that was processed
            success: Whether page processing was successful
        """
        status = self.file_handler.load_status(job_id)

        if not status:
            logger.error(f"No status found for job {job_id}")
            return

        status.setdefault("search_indexed", False)
        status.setdefault("search_index_status", "pending")
        status.setdefault("search_index_started_at", None)
        status.setdefault("search_indexed_at", None)
        status.setdefault("search_index_error", None)
        status.setdefault("search_index_version", None)
        status.setdefault("search_chunk_count", 0)
        status.setdefault("search_expected_chunk_count", 0)
        status.setdefault("search_saved_chunk_count", 0)
        status.setdefault("search_embedded_chunk_count", 0)
        status.setdefault("search_embedding_failed_count", 0)
        status.setdefault("search_index_health", "pending")

        # Update processed or failed pages
        if success:
            if page_num not in status["processed_pages"]:
                status["processed_pages"].append(page_num)
            if page_num in status["failed_pages"]:
                status["failed_pages"].remove(page_num)
        else:
            if page_num not in status["failed_pages"]:
                status["failed_pages"].append(page_num)
            if page_num in status["processed_pages"]:
                status["processed_pages"].remove(page_num)

        # Calculate percentage
        total_processed = len(status["processed_pages"]) + len(status["failed_pages"])
        total_pages = status["total_pages"]
        status["percentage"] = (
            round(float(total_processed) / total_pages * 100, 2)
            if total_pages > 0
            else 0.0
        )

        # Update status based on progress
        if total_processed >= total_pages:
            if len(status["failed_pages"]) == 0:
                status["status"] = "completed"
                status["completed_at"] = datetime.now().isoformat()
            else:
                status["status"] = "failed"
                status["error"] = (
                    f"{len(status['failed_pages'])} of {total_pages} pages failed"
                )
                status["completed_at"] = datetime.now().isoformat()

        self.file_handler.save_status(job_id, status)
        self._sync_to_db(job_id)

    def update_language(self, job_id: str, language: str) -> None:
        """
        Update job language after detection

        Args:
            job_id: Job identifier
            language: Detected language code
        """
        status = self.file_handler.load_status(job_id)
        if status:
            status["language"] = language
            self.file_handler.save_status(job_id, status)
            self._sync_to_db(job_id)

    def update_doc_type(self, job_id: str, doc_type: str) -> None:
        """
        Update job doc_type after detection

        Args:
            job_id: Job identifier
            doc_type: Detected document type ('scanned' or 'native')
        """
        status = self.file_handler.load_status(job_id)
        if status:
            status["doc_type"] = doc_type
            self.file_handler.save_status(job_id, status)
            self._sync_to_db(job_id)

    def mark_failed(self, job_id: str, error_message: str) -> None:
        """
        Mark job as failed

        Args:
            job_id: Job identifier
            error_message: Error message
        """
        status = self.file_handler.load_status(job_id)

        if status:
            status["status"] = "failed"
            status["error"] = error_message
            status["completed_at"] = datetime.now().isoformat()
            self.file_handler.save_status(job_id, status)
            self._sync_to_db(job_id)
            self.file_handler.log_error(job_id, f"Job failed: {error_message}")

    def mark_indexing_started(self, job_id: str, index_version: int) -> None:
        status = self.file_handler.load_status(job_id)
        if status:
            status["search_indexed"] = False
            status["search_index_status"] = "indexing"
            status["search_index_health"] = "indexing"
            status["search_index_started_at"] = datetime.now().isoformat()
            status["search_index_error"] = None
            status["search_index_version"] = index_version
            self.file_handler.save_status(job_id, status)
            self._sync_to_db(job_id)

    def mark_indexing_complete(
        self,
        job_id: str,
        *,
        chunk_count: int,
        index_version: int,
        expected_chunk_count: int | None = None,
        saved_chunk_count: int | None = None,
        embedded_chunk_count: int | None = None,
        embedding_failed_count: int | None = None,
        index_health: str = "healthy",
    ) -> None:
        status = self.file_handler.load_status(job_id)
        if status:
            status["search_indexed"] = True
            status["search_index_status"] = "indexed" if index_health == "healthy" else index_health
            status["search_indexed_at"] = datetime.now().isoformat()
            status["search_index_error"] = None
            status["search_index_version"] = index_version
            status["search_chunk_count"] = chunk_count
            status["search_expected_chunk_count"] = expected_chunk_count if expected_chunk_count is not None else chunk_count
            status["search_saved_chunk_count"] = saved_chunk_count if saved_chunk_count is not None else chunk_count
            status["search_embedded_chunk_count"] = embedded_chunk_count if embedded_chunk_count is not None else chunk_count
            status["search_embedding_failed_count"] = embedding_failed_count if embedding_failed_count is not None else 0
            status["search_index_health"] = index_health
            self.file_handler.save_status(job_id, status)
            self._sync_to_db(job_id)

    def mark_pending(self, job_id: str) -> None:
        """Persist a recovered job as queued and synchronize its metadata."""
        status = self.file_handler.load_status(job_id)
        if status:
            status["status"] = "pending"
            status["completed_at"] = None
            status["error"] = None
            self.file_handler.save_status(job_id, status)
            self._sync_to_db(job_id)

    def mark_indexing_failed(
        self, job_id: str, error_message: str, index_version: int | None = None
    ) -> None:
        status = self.file_handler.load_status(job_id)
        if status:
            status["search_indexed"] = False
            status["search_index_status"] = "failed"
            status["search_index_error"] = error_message
            status["search_index_health"] = "failed"
            if index_version is not None:
                status["search_index_version"] = index_version
            self.file_handler.save_status(job_id, status)
            self._sync_to_db(job_id)
            self.file_handler.log_error(
                job_id, f"Search indexing failed: {error_message}"
            )

    def _sync_to_db(self, job_id: str) -> None:
        """Trigger background sync of metadata to MongoDB."""
        summary = self.get_summary(job_id)
        if summary.get("status") == "not_found":
            return
            
        try:
            import asyncio
            from db.mongo import get_db
            from db.job_repo import upsert_job_meta
            
            # Use counts for mirroring
            summary["processed_pages_count"] = summary.get("processed_pages", 0)
            summary["failed_pages_count"] = summary.get("failed_pages", 0)

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(upsert_job_meta(get_db(), job_id, summary))
            except RuntimeError:
                # No running loop, try to use JobManager's captured loop
                from core.job_manager import get_job_manager
                manager = get_job_manager()
                if manager.loop and manager.loop.is_running():
                    manager.loop.call_soon_threadsafe(
                        lambda: asyncio.create_task(upsert_job_meta(get_db(), job_id, summary))
                    )
        except Exception as e:
            # Failure here should NOT stop the OCR process
            logger.debug(f"Metadata sync scheduled: {e}")


    def get_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get current job status

        Args:
            job_id: Job identifier

        Returns:
            Status dictionary
        """
        status = self.file_handler.load_status(job_id)

        if not status:
            return {"status": "not_found", "error": f"Job {job_id} not found"}

        return status

    def get_summary(self, job_id: str) -> Dict[str, Any]:
        """
        Get job summary with statistics

        Args:
            job_id: Job identifier

        Returns:
            Summary dictionary
        """
        status = self.get_status(job_id)

        if status.get("status") == "not_found":
            return status

        total_pages = status.get("total_pages", 0)
        processed_pages = status.get("processed_pages", [])
        failed_pages = status.get("failed_pages", [])
        processed_count = processed_pages if isinstance(processed_pages, int) else len(processed_pages)
        failed_count = failed_pages if isinstance(failed_pages, int) else len(failed_pages)

        # Calculate time taken if available
        started_at = status.get("started_at")
        completed_at = status.get("completed_at")
        time_taken = None

        if started_at and completed_at:
            try:
                start_time = datetime.fromisoformat(started_at)
                end_time = datetime.fromisoformat(completed_at)
                duration = (end_time - start_time).total_seconds()
                time_taken = f"{int(duration // 60)}m {int(duration % 60)}s"
            except:
                pass
        elif started_at and status.get("status") == "running":
            # Live elapsed time for in-progress jobs
            try:
                start_time = datetime.fromisoformat(started_at)
                duration = (datetime.now() - start_time).total_seconds()
                time_taken = f"{int(duration // 60)}m {int(duration % 60)}s"
            except:
                pass

        return {
            "job_id": job_id,
            "status": status.get("status"),
            "pdf_filename": status.get("pdf_filename"),
            "engine": status.get("engine"),
            "total_pages": total_pages,
            "source_total_pages": status.get("source_total_pages", total_pages),
            "source_preview_error": status.get("source_preview_error", ""),
            "processed_pages": processed_count,
            "failed_pages": failed_count,
            "percentage": status.get("percentage", 0),
            "time_taken": time_taken,
            "created_at": status.get("created_at"),
            "started_at": started_at,
            "completed_at": completed_at,
            "error": status.get("error"),
            "search_indexed": status.get("search_indexed", False),
            "search_index_status": status.get("search_index_status", "pending"),
            "search_index_started_at": status.get("search_index_started_at"),
            "search_indexed_at": status.get("search_indexed_at"),
            "search_index_error": status.get("search_index_error"),
            "search_index_version": status.get("search_index_version"),
            "search_chunk_count": status.get("search_chunk_count", 0),
            "search_expected_chunk_count": status.get("search_expected_chunk_count", status.get("search_chunk_count", 0)),
            "search_saved_chunk_count": status.get("search_saved_chunk_count", status.get("search_chunk_count", 0)),
            "search_embedded_chunk_count": status.get("search_embedded_chunk_count", status.get("search_chunk_count", 0)),
            "search_embedding_failed_count": status.get("search_embedding_failed_count", 0),
            "search_index_health": status.get("search_index_health", "unknown"),
            "category": status.get("category", ""),
            "categories": status.get("categories") or ([status.get("category")] if status.get("category") else []),
            "tags": status.get("tags") or [],
            "sheet_name": status.get("sheet_name", ""),
            "doc_type": status.get("doc_type", ""),
        }
