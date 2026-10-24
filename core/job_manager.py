from pathlib import Path
from typing import Optional
import os
import uuid
import logging
import threading
import queue
from contextlib import nullcontext
import time
import concurrent.futures

from storage.file_handler import FileHandler
from storage.resume_handler import ResumeHandler
from core.progress_tracker import ProgressTracker
from core.worker import PDFWorker
from core.embeddings import get_embedding_engine
from core.image_preprocessor import PreprocessOptions
from core.semantic_search import build_search_chunks
from db.mongo import get_db
from db.doc_repo import count_chunks, save_document, save_chunks
from config import settings
import asyncio
from core.source_preview import SourcePreviewError, count_pdf_pages, create_office_pdf_preview

logger = logging.getLogger(__name__)


class JobManager:
    """Manage OCR jobs and background processing"""

    def __init__(self):
        self.storage = FileHandler(settings.output_dir)
        logger.info("Using Local File Storage")

        self.resume_handler = ResumeHandler(self.storage)
        self.progress_tracker = ProgressTracker(self.storage)
        self.upload_dir = Path(settings.upload_dir)
        self.upload_dir.mkdir(exist_ok=True)
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._processing_queue: queue.Queue[str] = queue.Queue()
        self._queued_job_ids: set[str] = set()
        self._queue_lock = threading.Lock()
        self._queue_started = False
        self._indexing_futures: dict[str, concurrent.futures.Future] = {}
        self._indexing_lock = threading.Lock()
        
        # Trigger background sync of all existing jobs to MongoDB
        threading.Thread(target=self.sync_metadata_index, daemon=True).start()

    def sync_metadata_index(self) -> None:
        """Crawl filesystem and ensure MongoDB jobs collection is populated."""
        import asyncio
        from db.mongo import get_db
        from db.job_repo import upsert_job_meta
        
        logger.info("Starting background metadata sync...")
        job_ids = self.storage.list_jobs()
        count = 0
        
        for job_id in job_ids:
            try:
                summary = self.progress_tracker.get_summary(job_id)
                if summary.get("status") == "not_found":
                    continue
                
                summary["processed_pages_count"] = summary.get("processed_pages", 0)
                summary["failed_pages_count"] = summary.get("failed_pages", 0)

                async def do_upsert():
                    await upsert_job_meta(get_db(), job_id, summary)

                asyncio.run(do_upsert())
                count += 1
            except Exception as e:
                logger.debug(f"Failed to sync job {job_id}: {e}")
        
        logger.info(f"Metadata sync complete. Indexed {count} jobs.")

    def create_job(
        self, pdf_path: Path, pdf_filename: str, ocr_lang: str = "en", doc_type: str = "auto", category: str = "", categories: list[str] | None = None, tags: list[str] | None = None, client_upload_id: str = ""
    ) -> str:
        """
        Create a new OCR job

        Args:
            pdf_path: Path to uploaded PDF
            pdf_filename: Original filename
            ocr_lang: Language code for OCR

        Returns:
            Job ID
        """
        job_id = str(uuid.uuid4())

        self.storage.create_job_directory(job_id)

        if hasattr(self.storage, "save_source_pdf"):
            self.storage.save_source_pdf(job_id, pdf_path)

        import fitz

        try:
            pdf_doc = fitz.open(pdf_path)
            total_pages = len(pdf_doc)
            pdf_doc.close()
        except Exception as e:
            logger.error(f"Failed to open PDF: {e}")
            raise ValueError(f"Invalid PDF file: {str(e)}")

        source_total_pages = count_pdf_pages(self.storage.get_source_pdf_path(job_id)) if hasattr(self.storage, "get_source_pdf_path") else total_pages

        self.progress_tracker.initialize_job(
            job_id=job_id,
            total_pages=total_pages,
            pdf_filename=pdf_filename,
            engine=settings.ocr_engine,
            language=ocr_lang,
            doc_type=doc_type,
            category=category,
            categories=categories,
            tags=tags,
            source_total_pages=source_total_pages,
            source_extension="pdf",
            client_upload_id=client_upload_id,
        )

        logger.info(
            f"Created job {job_id} for {pdf_filename} ({total_pages} pages, lang: {ocr_lang})"
        )

        return job_id

    def start_processing(self, job_id: str, pdf_path: Path) -> None:
        """Queue a PDF job for durable single-worker processing."""
        self.enqueue_job(job_id)

    # ── Excel jobs ─────────────────────────────────────────────────────────

    def create_excel_job(
        self,
        xlsx_path: Path,
        xlsx_filename: str,
        sheet_name: str,
        category: str = "",
        categories: list[str] | None = None,
        tags: list[str] | None = None,
        client_upload_id: str = "",
    ) -> str:
        """Create a job record for an Excel sheet extraction (1 virtual page)."""
        job_id = str(uuid.uuid4())
        self.storage.create_job_directory(job_id)
        self.storage.save_source_file(job_id, xlsx_path)
        source_total_pages = 1
        source_preview_error = ""

        self.progress_tracker.initialize_job(
            job_id=job_id,
            total_pages=1,
            pdf_filename=xlsx_filename,
            engine="excel",
            language="auto",
            doc_type="native",
            category=category,
            categories=categories,
            tags=tags,
            sheet_name=sheet_name,
            source_total_pages=source_total_pages,
            source_preview_error=source_preview_error,
            source_extension="xlsx",
            client_upload_id=client_upload_id,
        )
        logger.info("Created Excel job %s for %s sheet=%r", job_id, xlsx_filename, sheet_name)
        return job_id

    def start_excel_processing(self, job_id: str, xlsx_path: Path, sheet_name: str) -> None:
        """Queue an Excel job for durable single-worker processing."""
        self.enqueue_job(job_id)

    # ── Word jobs ──────────────────────────────────────────────────────────

    def create_word_job(
        self,
        docx_path: Path,
        docx_filename: str,
        category: str = "",
        categories: list[str] | None = None,
        tags: list[str] | None = None,
        client_upload_id: str = "",
    ) -> str:
        """Create a job record for a native Word document extraction."""
        job_id = str(uuid.uuid4())
        self.storage.create_job_directory(job_id)
        self.storage.save_source_file(job_id, docx_path)
        source_total_pages = 1
        source_preview_error = ""
        if hasattr(self.storage, "get_source_pdf_path"):
            try:
                _, source_total_pages = create_office_pdf_preview(
                    docx_path,
                    self.storage.get_source_pdf_path(job_id),
                )
            except SourcePreviewError as exc:
                source_preview_error = str(exc)
                logger.warning("Word source preview unavailable for job %s: %s", job_id, exc)

        self.progress_tracker.initialize_job(
            job_id=job_id,
            total_pages=source_total_pages,
            pdf_filename=docx_filename,
            engine="word",
            language="auto",
            doc_type="native",
            category=category,
            categories=categories,
            tags=tags,
            source_total_pages=source_total_pages,
            source_preview_error=source_preview_error,
            source_extension="docx",
            client_upload_id=client_upload_id,
        )
        logger.info("Created Word job %s for %s", job_id, docx_filename)
        return job_id

    def start_word_processing(self, job_id: str, docx_path: Path) -> None:
        """Queue a Word job for durable single-worker processing."""
        self.enqueue_job(job_id)

    # ── Durable processing queue ──────────────────────────────────────────

    def start_queue(self) -> None:
        """Start the single consumer and recover jobs interrupted by a restart."""
        with self._queue_lock:
            if self._queue_started:
                return
            self._queue_started = True
            thread = threading.Thread(
                target=self._queue_worker,
                daemon=True,
                name="document-processing-queue",
            )
            thread.start()
        removed = self.cleanup_stale_transfers()
        recovered = self.recover_interrupted_jobs()
        logger.info(
            "Durable processing queue started; recovered %d job(s), removed %d stale transfer(s)",
            recovered,
            removed,
        )

    def cleanup_stale_transfers(self, max_age_seconds: int = 24 * 60 * 60) -> int:
        """Remove abandoned unique transfer files without touching legacy sources."""
        removed = 0
        cutoff = time.time() - max_age_seconds
        for path in self.upload_dir.iterdir():
            name = path.name
            is_partial = name.startswith(".") and name.endswith(".part")
            is_finalized_transfer = (
                len(name) > 33
                and name[32] == "-"
                and all(char in "0123456789abcdef" for char in name[:32].lower())
            )
            try:
                if path.is_file() and (is_partial or is_finalized_transfer) and path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except OSError as exc:
                logger.warning("Could not remove stale transfer %s: %s", path, exc)
        return removed

    def enqueue_job(self, job_id: str) -> bool:
        """Idempotently place a persisted job on the in-process work queue."""
        if not self.storage.job_exists(job_id):
            logger.error("Cannot queue missing job %s", job_id)
            return False
        with self._queue_lock:
            if job_id in self._queued_job_ids:
                return False
            self._queued_job_ids.add(job_id)
            self._processing_queue.put(job_id)
        logger.info("Queued job %s", job_id)
        return True

    def recover_interrupted_jobs(self) -> int:
        """Requeue persisted pending/running jobs and restart incomplete indexing."""
        recovered = 0
        processing_candidates: list[tuple[str, int, str]] = []
        for discovery_order, job_id in enumerate(self.storage.list_jobs()):
            status_info = self.progress_tracker.get_status(job_id)
            state = status_info.get("status")
            if state in {"pending", "running"}:
                processing_candidates.append((
                    str(status_info.get("created_at") or ""),
                    discovery_order,
                    job_id,
                ))
            elif state == "completed" and status_info.get("search_index_status") == "indexing":
                # Do not start invisible embedding work merely because the server
                # restarted. Keep the OCR result completed and require an explicit
                # reindex request for work that was interrupted during indexing.
                self.progress_tracker.mark_indexing_failed(
                    job_id,
                    "Search indexing was interrupted by a server restart; manual reindex required",
                    getattr(settings, "search_index_version", 1),
                )
                logger.info(
                    "Marked interrupted semantic indexing for job %s as failed; not auto-resuming",
                    job_id,
                )
        for _, _, job_id in sorted(processing_candidates):
            self.progress_tracker.mark_pending(job_id)
            recovered += int(self.enqueue_job(job_id))
        return recovered

    def _schedule_index_recovery(self, job_id: str) -> None:
        self.schedule_indexing(job_id, force=True)

    def schedule_indexing(
        self,
        job_id: str,
        filename: Optional[str] = None,
        force: bool = True,
    ) -> bool:
        """Schedule one tracked indexing task, deduplicated by job ID."""
        if not self.loop or not self.loop.is_running():
            return False
        with self._indexing_lock:
            existing = self._indexing_futures.get(job_id)
            if existing and not existing.done():
                logger.info("Indexing already scheduled for job %s", job_id)
                return True
            future = asyncio.run_coroutine_threadsafe(
                self.index_job(job_id, filename=filename, force=force),
                self.loop,
            )
            self._indexing_futures[job_id] = future
        future.add_done_callback(
            lambda fut, jid=job_id: self._finish_indexing_future(jid, fut)
        )
        logger.info("Scheduled tracked semantic indexing for job %s", job_id)
        return True

    def _finish_indexing_future(self, job_id: str, future) -> None:
        with self._indexing_lock:
            if self._indexing_futures.get(job_id) is future:
                self._indexing_futures.pop(job_id, None)
        self._log_indexing_future(job_id, future)

    def _queue_worker(self) -> None:
        while True:
            job_id = self._processing_queue.get()
            try:
                process_lock = getattr(self.storage, "processing_lock", None)
                with (process_lock() if process_lock else nullcontext()):
                    self._process_queued_job(job_id)
            except Exception as exc:
                logger.error("Queued job %s failed: %s", job_id, exc, exc_info=True)
                self.progress_tracker.mark_failed(job_id, f"Queued processing failed: {exc}")
            finally:
                with self._queue_lock:
                    self._queued_job_ids.discard(job_id)
                self._processing_queue.task_done()

    def _process_queued_job(self, job_id: str) -> None:
        status_info = self.progress_tracker.get_status(job_id)
        if status_info.get("status") not in {"pending", "running"}:
            logger.info(
                "Skipping stale queue entry for job %s (status=%s)",
                job_id,
                status_info.get("status"),
            )
            return
        extension = self._source_extension(status_info)
        source_path = self.resolve_source_path(job_id)
        logger.info("Processing queued %s job %s from %s", extension, job_id, source_path)

        if extension == "docx":
            from core.word_worker import WordWorker
            WordWorker(
                job_id=job_id,
                docx_path=source_path,
                storage=self.storage,
                progress_tracker=self.progress_tracker,
                resume_handler=self.resume_handler,
                job_manager=self,
            ).process()
        elif extension == "xlsx":
            from core.excel_worker import ExcelWorker
            ExcelWorker(
                job_id=job_id,
                xlsx_path=source_path,
                sheet_name=status_info.get("sheet_name") or "",
                storage=self.storage,
                progress_tracker=self.progress_tracker,
                resume_handler=self.resume_handler,
                job_manager=self,
            ).process()
        else:
            job_lang = status_info.get("language", settings.ocr_lang)
            self._run_job_process(
                job_id,
                source_path,
                status_info.get("engine") or settings.ocr_engine,
                self._build_engine_kwargs(job_lang),
            )

    def resolve_source_path(self, job_id: str) -> Path:
        """Resolve the immutable job-scoped source, with legacy fallback."""
        status_info = self.progress_tracker.get_status(job_id)
        extension = self._source_extension(status_info)
        source_path = self.storage.get_source_file_path(job_id, extension)
        if source_path.exists():
            return source_path

        legacy_path = self.upload_dir / str(status_info.get("pdf_filename") or "")
        if legacy_path.exists():
            return self.storage.save_source_file(job_id, legacy_path)
        if extension == "pdf" and self.storage.download_source_pdf(job_id, source_path):
            return source_path
        raise FileNotFoundError(f"Original {extension.upper()} source for job {job_id} is unavailable")

    @staticmethod
    def _source_extension(status_info: dict) -> str:
        explicit = str(status_info.get("source_extension") or "").lower().lstrip(".")
        if explicit in {"pdf", "docx", "xlsx"}:
            return explicit
        engine = str(status_info.get("engine") or "").lower()
        return "docx" if engine == "word" else "xlsx" if engine == "excel" else "pdf"

    def find_job_by_client_upload_id(self, client_upload_id: str) -> Optional[dict]:
        """Return an existing job summary for an idempotent client retry."""
        if not client_upload_id:
            return None
        for job_id in self.storage.list_jobs():
            status_info = self.progress_tracker.get_status(job_id)
            if status_info.get("client_upload_id") == client_upload_id:
                return status_info
        return None


    @staticmethod
    def _build_engine_kwargs(job_lang: str) -> dict:
        """Return engine kwargs for the configured OCR engine.

        The ``preprocess_options`` value is built from the live
        :class:`config.Settings` object and tuned for the *job's* language
        (Tamil pages get CLAHE + binarisation off; Latin pages keep their
        defaults). The dataclass is frozen so it is pickle-safe to send
        across the worker-process boundary.

        ``"auto"`` is intentionally forwarded to the worker so that
        :meth:`core.worker.PDFWorker._maybe_autodetect_language` can probe
        the document and select the correct language.  ``"auto"`` is only
        resolved to ``"en"`` for ``PreprocessOptions`` (which needs a concrete
        language code) and at the PaddleOCR call site as a safety net.
        """
        # "auto" is valid for the worker pipeline but not for PreprocessOptions.
        preprocess_lang = job_lang if job_lang and job_lang != "auto" else "en"
        base_options = PreprocessOptions.from_settings(settings)
        preprocess_options = base_options.for_language(preprocess_lang)
        common = {
            "lang": job_lang,  # may be "auto"; worker resolves via auto-detection
            "use_gpu": settings.is_gpu_enabled(),
            "enable_translation": settings.enable_translation,
            "translation_target_lang": settings.translation_target_lang,
            "preprocess_options": preprocess_options,
        }
        if settings.ocr_engine == "paddle":
            return {
                **common,
                "use_layout": settings.use_layout_detection,
                "layout_recovery": settings.layout_recovery,
                "table_as_html": settings.layout_table_as_html,
            }
        if settings.ocr_engine == "remote":
            return {
                **common,
                "use_layout": settings.use_layout_detection,
                "base_url": settings.ocr_remote_url,
                "auth": settings.ocr_remote_auth,
                "api_key": settings.ocr_remote_api_key,
                "model": settings.ocr_remote_model,
                "timeout_seconds": settings.ocr_remote_timeout_seconds,
                "verify_tls": settings.httpx_verify(),
            }
        if settings.ocr_engine == "google_docai":
            return {
                **common,
                "use_layout": settings.use_layout_detection,
                "credentials_path": settings.google_docai_credentials,
                "processor_name": settings.docai_processor_resource_name(),
                "processor_version": settings.google_docai_processor_version,
                "language_hints": settings.google_docai_language_hints,
                "timeout_seconds": settings.google_docai_timeout,
                # Production quality knobs
                "native_pdf": settings.google_docai_native_pdf,
                "enable_native_pdf_parsing": settings.google_docai_enable_native_pdf_parsing,
                "compute_style_info": settings.google_docai_compute_style_info,
                "min_confidence": settings.google_docai_min_confidence,
                "max_retries": settings.google_docai_max_retries,
            }
        if settings.ocr_engine in ("paddle_vl", "PaddleOCR1.5"):
            # Remote API mode: delegate VLM inference to deployed microservice
            if settings.paddle_vl_api_url:
                return {
                    "base_url": settings.paddle_vl_api_url,
                    "api_key": settings.paddle_vl_api_key,
                    "timeout_seconds": settings.paddle_vl_api_timeout,
                    # use_gpu is irrelevant for the remote engine but the
                    # worker checks it for the parallel-worker guard.
                    "use_gpu": False,
                }
            # Local VLM mode: load model in-process (requires GPU / high RAM)
            return {
                "use_gpu": settings.is_gpu_enabled(),
                "load_in_8bit": settings.paddle_vl_load_in_8bit,
                "load_in_4bit": settings.paddle_vl_load_in_4bit,
                "max_new_tokens": settings.paddle_vl_max_new_tokens,
                "model_cache_dir": settings.paddle_vl_cache_dir or None,
            }
        if settings.ocr_engine == "olm_ocr":
            return {
                "use_gpu": settings.is_gpu_enabled(),
                "provider": settings.olm_ocr_provider,
                "model_id": settings.olm_ocr_model_id,
                "load_in_8bit": settings.olm_ocr_load_in_8bit,
                "load_in_4bit": settings.olm_ocr_load_in_4bit,
                "max_new_tokens": settings.olm_ocr_max_new_tokens,
                "model_cache_dir": settings.olm_ocr_path or None,
                "temperature": settings.olm_ocr_temperature,
                "openai_api_key": settings.openai_api_key,
                "api_model": settings.olm_ocr_api_model,
            }
        if settings.ocr_engine == "google_visionai":
            return {
                **common,
                "use_layout": settings.use_layout_detection,
                "credentials_path": settings.google_visionai_credentials,
                "language_hints": settings.google_visionai_language_hints,
                "timeout_seconds": settings.google_visionai_timeout,
                "max_retries": settings.google_visionai_max_retries,
            }
        if settings.ocr_engine in ("sur_ocr", "ocr_open_ai"):
            return {
                **common,
                "use_layout": settings.use_layout_detection,
            }
        return common

    def _run_job_process(
        self, job_id: str, pdf_path: Path, engine_type: str, engine_kwargs: dict
    ) -> None:
        """
        Target function for the background thread.
        Runs in the same process as the FastAPI server — shares storage/trackers directly.
        The inner ProcessPoolExecutor inside PDFWorker spawns clean child processes
        (one level only), which works correctly on Windows.
        """
        try:
            status_info = self.progress_tracker.get_status(job_id)
            worker = PDFWorker(
                job_id=job_id,
                pdf_path=pdf_path,
                engine_type=engine_type,
                engine_kwargs=engine_kwargs,
                storage=self.storage,
                progress_tracker=self.progress_tracker,
                resume_handler=self.resume_handler,
                max_workers=settings.max_workers,
                doc_type=status_info.get("doc_type", "auto")
            )
            worker.process_pdf()

            # Post-processing: Schedule semantic search indexing on the main event loop.
            # We fire-and-forget — blocking here would stall the OCR thread for
            # minutes while embeddings are generated for hundreds of chunks.
            try:
                status = self.progress_tracker.get_status(job_id)
                filename = status.get("pdf_filename", pdf_path.name)

                if self.loop and self.loop.is_running():
                    logger.info("Scheduling semantic indexing for job %s", job_id)
                    self.schedule_indexing(job_id, filename=filename, force=True)
                else:
                    logger.warning(
                        "Main event loop not available; running indexing synchronously for %s",
                        job_id,
                    )
                    asyncio.run(self.index_job(job_id, filename=filename, force=True))
            except Exception as e:
                logger.error(
                    "Failed to schedule indexing for job %s: %s",
                    job_id, e, exc_info=True,
                )
        except Exception as e:
            error_msg = f"Background thread for job {job_id} failed: {e}"
            logger.error(error_msg, exc_info=True)
            self.progress_tracker.mark_failed(job_id, error_msg)

    def _log_indexing_future(self, job_id: str, future) -> None:
        """Log background indexing failures that would otherwise be invisible."""
        try:
            result = future.result()
            logger.info(
                "Semantic indexing finished for job %s (indexed=%s chunks=%s)",
                job_id,
                result.get("indexed"),
                result.get("chunk_count"),
            )
        except concurrent.futures.CancelledError:
            logger.info("Semantic indexing cancelled for job %s", job_id)
        except Exception as exc:
            logger.error(
                "Semantic indexing task failed for job %s: %s",
                job_id,
                exc,
                exc_info=True,
            )

    async def index_job(
        self, job_id: str, filename: Optional[str] = None, force: bool = False
    ) -> dict:
        """Build or rebuild the MongoDB search index for a job."""
        status = self.progress_tracker.get_status(job_id)
        if status.get("status") == "not_found":
            raise ValueError(f"Job {job_id} not found")

        index_version = getattr(settings, "search_index_version", 1)
        if (
            not force
            and status.get("search_indexed")
            and status.get("search_index_version") == index_version
        ):
            existing_chunks = await count_chunks(get_db(), job_id)
            if existing_chunks > 0:
                return {
                    "job_id": job_id,
                    "indexed": True,
                    "skipped": True,
                    "chunk_count": existing_chunks,
                    "index_version": index_version,
                }
            logger.warning(
                "Job %s was marked indexed but has 0 Mongo chunks; rebuilding index.",
                job_id,
            )

        resolved_filename = filename or status.get("pdf_filename") or f"{job_id}.pdf"
        resolved_category = (status.get("category") or "").strip()
        resolved_categories = status.get("categories") or ([resolved_category] if resolved_category else [])
        self.progress_tracker.mark_indexing_started(job_id, index_version)

        text = self.storage.read_result(job_id)
        if not text:
            message = f"No text content found for job {job_id}, skipping DB storage."
            logger.warning(message)
            self.progress_tracker.mark_indexing_failed(job_id, message, index_version)
            return {
                "job_id": job_id,
                "indexed": False,
                "chunk_count": 0,
                "index_version": index_version,
                "error": message,
            }

        structured = self.storage.read_structured_result(job_id)
        from core.doc_metadata import extract_document_metadata
        doc_metadata = extract_document_metadata(text, resolved_filename)
        engine = get_embedding_engine()
        db = get_db()
        chunks = build_search_chunks(
            doc_id=job_id,
            filename=resolved_filename,
            text=text,
            structured=structured,
            chunk_size=getattr(settings, "search_chunk_size", 1200),
            overlap=getattr(settings, "search_chunk_overlap", 150),
        )

        try:
            # Generate document-level embedding (optional fallback)
            doc_embedding = []
            if text.strip():
                try:
                    doc_embedding = await engine.aget_embedding(text[:10000])
                except Exception as emb_e:
                    logger.warning(f"Doc embedding failed (ignoring): {emb_e}")

            await save_document(
                db=db,
                job_id=job_id,
                filename=resolved_filename,
                content=text,
                embedding=doc_embedding,
                metadata={
                    "index_version": index_version,
                    "chunk_count": len(chunks),
                    "expected_chunk_count": len(chunks),
                    "structured_available": bool(
                        structured and structured.get("pages")
                    ),
                    "category": resolved_category,
                    "categories": resolved_categories,
                    **doc_metadata,
                },
            )

            embed_parent = getattr(settings, "embed_parent_chunks", False)
            eligible_chunks = [
                chunk for chunk in chunks
                if (chunk.metadata.get("chunk_type") != "parent" or embed_parent) and len(chunk.text.strip()) > 20
            ]
            parent_chunks = [
                chunk for chunk in chunks
                if chunk.metadata.get("chunk_type") == "parent" and not embed_parent
            ]
            # Batch embeddings (in-process is free; remote batch ~6× faster than N POSTs).
            batch_size = max(
                1,
                min(
                    int(getattr(settings, "search_embedding_concurrency", 4)) * 8,
                    8 if engine.backend == "remote" else 64,
                ),
            )
            retries = max(1, int(getattr(settings, "search_embedding_retries", 3)))
            embed_results = []
            for start in range(0, len(eligible_chunks), batch_size):
                batch = eligible_chunks[start : start + batch_size]
                inputs = [(c.embed_text or c.text) for c in batch]
                vectors: list = []
                for attempt in range(retries):
                    try:
                        vectors = await engine.aget_embeddings(inputs, db=db)
                        if vectors and any(vectors):
                            break
                    except Exception as batch_exc:
                        logger.warning(
                            "Chunk embedding batch failed (attempt %d): %s",
                            attempt + 1,
                            batch_exc,
                        )
                        vectors = []
                    if attempt < retries - 1:
                        await asyncio.sleep(0.35 * (attempt + 1))
                if len(vectors) < len(batch):
                    vectors = list(vectors) + [[] for _ in range(len(batch) - len(vectors))]
                for chunk, embedding, emb_input in zip(batch, vectors, inputs):
                    embed_results.append((chunk, embedding or [], len(emb_input)))

            chunk_docs = []
            embedding_failed_count = 0
            embedded_count = 0
            for item in embed_results:
                if isinstance(item, Exception):
                    embedding_failed_count += 1
                    logger.warning(
                        "Chunk embedding failed with exception: %s",
                        item,
                    )
                    continue

                chunk, embedding, embedding_text_len = item
                embedding_status = "ok" if embedding else "failed"
                if not embedding:
                    embedding_failed_count += 1
                    logger.warning(
                        "Chunk %d (len=%d) saved for lexical search, but embedding returned empty. "
                        "Check EMBEDDING_RUNTIME / EMBEDDING_LOCAL_MODEL_PATH "
                        "(or EMBEDDING_MODEL_URL fallback) and the embedding backend.",
                        chunk.chunk_id,
                        len(chunk.text),
                    )
                else:
                    embedded_count += 1

                chunk_metadata = {
                    **chunk.metadata,
                    "emb_dim": len(embedding),
                    "embedding_text_len": embedding_text_len,
                    "embedding_strategy": "heading_page_context_v2",
                    "embedding_status": embedding_status,
                    "category": resolved_category,
                    "categories": resolved_categories,
                    "doc_classification": doc_metadata.get("doc_classification"),
                    "detected_language": doc_metadata.get("detected_language") or chunk.metadata.get("language"),
                    "doc_type": status.get("doc_type", "auto"),
                    "title": doc_metadata.get("extracted_title") or chunk.metadata.get("title"),
                    "dates": doc_metadata.get("extracted_dates"),
                    "parties": doc_metadata.get("parties"),
                }

                chunk_docs.append(
                    {
                        "filename": resolved_filename,
                        "chunk_id": chunk.chunk_id,
                        "text": chunk.text,
                        "embedding": embedding,
                        "metadata": chunk_metadata,
                        "file_path": chunk.file_path,
                        "index_version": index_version,
                    }
                )

            # Add parent chunks without embedding overhead
            for p_chunk in parent_chunks:
                p_metadata = {
                    **p_chunk.metadata,
                    "emb_dim": 0,
                    "embedding_text_len": len(p_chunk.text),
                    "embedding_strategy": "parent_chunk",
                    "embedding_status": "parent",
                    "category": resolved_category,
                    "categories": resolved_categories,
                    "doc_classification": doc_metadata.get("doc_classification"),
                    "detected_language": doc_metadata.get("detected_language") or p_chunk.metadata.get("language"),
                    "doc_type": status.get("doc_type", "auto"),
                    "title": doc_metadata.get("extracted_title") or p_chunk.metadata.get("title"),
                }
                chunk_docs.append(
                    {
                        "filename": resolved_filename,
                        "chunk_id": p_chunk.chunk_id,
                        "text": p_chunk.text,
                        "embedding": [],
                        "metadata": p_metadata,
                        "file_path": p_chunk.file_path,
                        "index_version": index_version,
                    }
                )


            if embedding_failed_count:
                logger.warning(
                    "job %s: %d/%d chunks have failed embeddings but remain lexically searchable.",
                    job_id, embedding_failed_count, len(eligible_chunks),
                )

            if eligible_chunks and not chunk_docs:
                message = (
                    "No usable text chunks were produced for search indexing."
                )
                self.progress_tracker.mark_indexing_failed(
                    job_id, message, index_version
                )
                return {
                    "job_id": job_id,
                    "indexed": False,
                    "chunk_count": 0,
                    "index_version": index_version,
                    "error": message,
                }

            success_ratio = (
                embedded_count / max(1, len(eligible_chunks))
                if eligible_chunks else 1.0
            )
            min_ratio = float(getattr(settings, "search_index_min_success_ratio", 0.95))
            if embedded_count == 0 and chunk_docs:
                index_health = "lexical_only"
            elif success_ratio >= min_ratio:
                index_health = "healthy"
            else:
                index_health = "degraded"

            for doc in chunk_docs:
                doc_metadata = doc.setdefault("metadata", {})
                doc_metadata["expected_chunk_count"] = len(eligible_chunks)
                doc_metadata["saved_chunk_count"] = len(chunk_docs)
                doc_metadata["embedded_chunk_count"] = embedded_count
                doc_metadata["embedding_failed_count"] = embedding_failed_count
                doc_metadata["index_health"] = index_health

            existing_health = status.get("search_index_health") or (
                "healthy" if status.get("search_indexed") else ""
            )
            existing_chunks = await count_chunks(db, job_id)
            if existing_health == "healthy" and index_health != "healthy" and existing_chunks > 0:
                message = (
                    f"Rebuild produced {index_health} index "
                    f"({embedded_count}/{len(eligible_chunks)} embedded); preserving existing healthy index."
                )
                logger.warning("Job %s: %s", job_id, message)
                self.progress_tracker.mark_indexing_failed(job_id, message, index_version)
                return {
                    "job_id": job_id,
                    "indexed": False,
                    "preserved_existing": True,
                    "chunk_count": existing_chunks,
                    "expected_chunk_count": len(eligible_chunks),
                    "embedded_chunk_count": embedded_count,
                    "embedding_failed_count": embedding_failed_count,
                    "index_health": "preserved_healthy",
                    "index_version": index_version,
                    "error": message,
                }

            await save_document(
                db=db,
                job_id=job_id,
                filename=resolved_filename,
                content=text,
                embedding=doc_embedding,
                metadata={
                    "index_version": index_version,
                    "chunk_count": len(chunk_docs),
                    "expected_chunk_count": len(eligible_chunks),
                    "saved_chunk_count": len(chunk_docs),
                    "embedded_chunk_count": embedded_count,
                    "embedding_failed_count": embedding_failed_count,
                    "index_health": index_health,
                    "structured_available": bool(
                        structured and structured.get("pages")
                    ),
                    "category": resolved_category,
                    "categories": resolved_categories,
                    **doc_metadata,
                },
            )

            await save_chunks(db, job_id, chunk_docs, replace_existing=True)
            self.progress_tracker.mark_indexing_complete(
                job_id,
                chunk_count=len(chunk_docs),
                index_version=index_version,
                expected_chunk_count=len(eligible_chunks),
                saved_chunk_count=len(chunk_docs),
                embedded_chunk_count=embedded_count,
                embedding_failed_count=embedding_failed_count,
                index_health=index_health,
            )
            logger.info(
                "Saved %d searchable chunks to MongoDB for job %s (embedded=%d failed=%d health=%s)",
                len(chunk_docs),
                job_id,
                embedded_count,
                embedding_failed_count,
                index_health,
            )
            return {
                "job_id": job_id,
                "indexed": True,
                "chunk_count": len(chunk_docs),
                "expected_chunk_count": len(eligible_chunks),
                "saved_chunk_count": len(chunk_docs),
                "embedded_chunk_count": embedded_count,
                "embedding_failed_count": embedding_failed_count,
                "index_health": index_health,
                "index_version": index_version,
            }
        except Exception as e:
            self.progress_tracker.mark_indexing_failed(job_id, str(e), index_version)
            raise

    async def smart_reindex(self, job_id: str, new_text: str, old_text: Optional[str] = None) -> dict:
        """
        Smart partial update: compares old_text vs new_text page by page.
        Only chunks and re-embeds pages that have changed.
        Supports structured layout-aware chunking if page JSONs are available.
        """
        from core.semantic_search import split_flat_text_pages, chunk_text, extract_section_label
        from core.doc_metadata import extract_document_metadata
        import json
        
        status = self.progress_tracker.get_status(job_id)
        if status.get("status") == "not_found":
            raise ValueError(f"Job {job_id} not found")

        if old_text is None:
            old_text = self.storage.read_result(job_id) or ""
        
        old_pages = dict(split_flat_text_pages(old_text))
        new_pages = dict(split_flat_text_pages(new_text))
        
        # Map None keys (single-page docs without page headers) to page 1
        if None in old_pages:
            old_pages[1] = old_pages.pop(None)
        if None in new_pages:
            new_pages[1] = new_pages.pop(None)
        
        changed_pages = {}
        for p_num, p_text in new_pages.items():
            if p_num not in old_pages or old_pages[p_num] != p_text:
                changed_pages[p_num] = p_text
                
        if not changed_pages:
            logger.info(f"Smart re-index for job {job_id}: No pages changed.")
            return {"job_id": job_id, "reindexed_pages": 0, "new_chunks": 0, "message": "No pages changed"}
            
        index_version = getattr(settings, "search_index_version", 1)
        resolved_filename = status.get("pdf_filename") or f"{job_id}.pdf"
        resolved_category = (status.get("category") or "").strip()
        resolved_categories = status.get("categories") or ([resolved_category] if resolved_category else [])
        
        engine = get_embedding_engine()
        db = get_db()
        
        chunk_size = getattr(settings, "search_chunk_size", 1200)
        overlap = getattr(settings, "search_chunk_overlap", 150)
        
        pages_dir = self.storage.get_job_directory(job_id) / "pages"
        total_new_chunks = 0
        
        for page_num, page_text in changed_pages.items():
            if db is not None:
                # Robust cleanup: for page 1/None, delete any old chunks with page=1, page=None, or missing page metadata
                if page_num == 1:
                    await db.chunks.delete_many({
                        "job_id": job_id,
                        "$or": [
                            {"metadata.page": 1},
                            {"metadata.page": None},
                            {"metadata.page": {"$exists": False}}
                        ]
                    })
                else:
                    await db.chunks.delete_many({"job_id": job_id, "metadata.page": page_num})
                
            page_file = pages_dir / f"page_{page_num:04d}.json"
            page_chunks = []
            
            if page_file.exists():
                try:
                    with open(page_file, "r", encoding="utf-8") as f:
                        page_data = json.load(f)
                    
                    element_idx = 0
                    for element in page_data.get("elements", []):
                        from core.semantic_search import normalize_ocr_text
                        element_text = normalize_ocr_text(element.get("text", ""))
                        if not element_text:
                            continue
                            
                        element_type = element.get("type", "paragraph")
                        base_metadata = {
                            "page": page_num,
                            "element_type": element_type,
                            "source": "structured",
                        }
                        
                        if element_type == "table":
                            page_chunks.append((element_text, base_metadata, element_idx))
                            element_idx += 1
                            continue
                            
                        for piece in chunk_text(element_text, chunk_size=chunk_size, overlap=overlap):
                            page_chunks.append((piece, base_metadata, element_idx))
                            element_idx += 1
                except Exception as e:
                    logger.error(f"Failed to read structured page {page_num} for job {job_id} reindexing, falling back to flat text chunking: {e}")
                    page_chunks = []
            
            if not page_chunks:
                # Fallback to flat text chunking
                raw_pieces = chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)
                for i, piece in enumerate(raw_pieces):
                    chunk_metadata = {
                        "page": page_num,
                        "source": "flat",
                    }
                    page_chunks.append((piece, chunk_metadata, i))
            
            chunk_docs = []
            for i, (piece, meta, inner_idx) in enumerate(page_chunks):
                if len(piece.strip()) <= 20: continue
                
                embedding = await engine.aget_embedding(piece)
                if not embedding: continue
                
                chunk_metadata = {
                    **meta,
                    "emb_dim": len(embedding),
                    "category": resolved_category,
                    "categories": resolved_categories,
                    "section_label": extract_section_label(piece)
                }
                
                base_chunk_id = (page_num or 0) * 10000 + inner_idx
                
                chunk_docs.append({
                    "job_id": job_id,
                    "filename": resolved_filename,
                    "chunk_id": base_chunk_id,
                    "text": piece,
                    "embedding": embedding,
                    "metadata": chunk_metadata,
                    "file_path": f"storage://{job_id}/result.json" if meta.get("source") == "structured" else f"storage://{job_id}/result.txt",
                    "index_version": index_version,
                })
                
            if chunk_docs and db is not None:
                await db.chunks.insert_many(chunk_docs)
                total_new_chunks += len(chunk_docs)
                
        logger.info(f"Smart re-index for job {job_id}: {len(changed_pages)} pages changed, {total_new_chunks} new chunks.")
        
        if db is not None:
            new_total = await count_chunks(db, job_id)
            self.progress_tracker.mark_indexing_complete(
                job_id,
                chunk_count=new_total,
                index_version=index_version,
            )
            
            # Keep MongoDB documents collection completely in sync
            doc_embedding = []
            if new_text.strip():
                try:
                    doc_embedding = await engine.aget_embedding(new_text[:10000])
                except Exception as emb_e:
                    logger.warning(f"Doc embedding update failed (ignoring): {emb_e}")
            
            structured = self.storage.read_structured_result(job_id)
            doc_metadata = extract_document_metadata(new_text, resolved_filename)
            await save_document(
                db=db,
                job_id=job_id,
                filename=resolved_filename,
                content=new_text,
                embedding=doc_embedding,
                metadata={
                    "index_version": index_version,
                    "chunk_count": new_total,
                    "structured_available": bool(
                        structured and structured.get("pages")
                    ),
                    "category": resolved_category,
                    "categories": resolved_categories,
                    **doc_metadata,
                },
            )
            
        return {
            "job_id": job_id, 
            "reindexed_pages": len(changed_pages),
            "new_chunks": total_new_chunks
        }

    async def backfill_search_indexes(self, force: bool = False) -> dict:
        """Index all completed jobs that are missing Mongo search chunks."""
        indexed_count = 0
        skipped_count = 0
        failed_count = 0
        jobs_list = []
        for job_id in self.storage.list_jobs():
            status = self.progress_tracker.get_status(job_id)
            if status.get("status") == "not_found" or status.get("status") != "completed":
                continue

            try:
                outcome = await self.index_job(job_id, force=force)
                if outcome.get("skipped"):
                    skipped_count += 1
                elif outcome.get("indexed"):
                    indexed_count += 1
                else:
                    failed_count += 1
                jobs_list.append(outcome)
            except Exception as e:
                failed_count += 1
                jobs_list.append(
                    {"job_id": job_id, "indexed": False, "error": str(e)}
                )

        return {
            "indexed": indexed_count,
            "skipped": skipped_count,
            "failed": failed_count,
            "jobs": jobs_list
        }

    def get_status(self, job_id: str) -> dict:
        """Get current status of a job"""
        return self.progress_tracker.get_summary(job_id)

    def get_result(self, job_id: str) -> Optional[str]:
        """Get job result text"""
        return self.storage.read_result(job_id)

    def resume_job(self, job_id: str, source_path: Path | None = None) -> bool:
        """
        Resume a failed or interrupted job

        Args:
            job_id: Job identifier
            source_path: Optional legacy source path. Job-scoped source is preferred.

        Returns:
            True if resume was initiated
        """
        if not self.storage.job_exists(job_id):
            logger.error(f"Job {job_id} does not exist")
            return False

        if not self.resume_handler.should_resume(job_id):
            logger.warning(f"Job {job_id} cannot be resumed")
            return False

        try:
            self.resolve_source_path(job_id)
        except FileNotFoundError:
            if source_path is None or not source_path.exists():
                logger.error("Source for resumable job %s is unavailable", job_id)
                return False
            self.storage.save_source_file(job_id, source_path)

        logger.info("Resuming job %s through durable queue", job_id)
        self.progress_tracker.mark_pending(job_id)
        if self.enqueue_job(job_id):
            return True
        with self._queue_lock:
            return job_id in self._queued_job_ids

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running job (mark as failed)

        Args:
            job_id: Job identifier

        Returns:
            True if job was cancelled
        """
        if not self.storage.job_exists(job_id):
            return False

        with self._indexing_lock:
            indexing_future = self._indexing_futures.pop(job_id, None)
        if indexing_future and not indexing_future.done():
            indexing_future.cancel()
            logger.info("Cancelled semantic indexing for job %s", job_id)
        current = self.progress_tracker.get_status(job_id)
        if current.get("status") == "completed":
            if current.get("search_index_status") == "indexing" or indexing_future:
                self.progress_tracker.mark_indexing_failed(
                    job_id,
                    "Search indexing cancelled by user",
                    getattr(settings, "search_index_version", 1),
                )
            else:
                return False
        else:
            self.progress_tracker.mark_failed(job_id, "Job cancelled by user")
        logger.info(f"Cancelled job {job_id}")
        return True

    def delete_job(self, job_id: str) -> bool:
        """
        Delete job and all associated files

        Args:
            job_id: Job identifier

        Returns:
            True if job was deleted
        """
        if not self.storage.job_exists(job_id):
            return False

        self.storage.cleanup_job(job_id)
        return True


# ---------------------------------------------------------------------------
# Module-level singleton — one instance shared across all API routes.
# This avoids re-initialising FileHandler (and the log noise) on every request.
# ---------------------------------------------------------------------------
_instance: "JobManager | None" = None
_instance_lock = threading.Lock()


def get_job_manager() -> "JobManager":
    """Return the shared JobManager singleton, creating it on first call."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:  # double-checked locking
                _instance = JobManager()
    return _instance
