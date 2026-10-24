import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path
from supabase import create_client, Client

logger = logging.getLogger(__name__)

class SupabaseHandler:
    """Handle file operations for job storage in Supabase Storage"""
    
    def __init__(self, settings):
        self.settings = settings
        self.bucket_name = settings.supabase_bucket
        self.supabase: Client = create_client(
            settings.supabase_url, 
            settings.supabase_service_key
        )
        
    def _get_path(self, job_id: str, filename: str) -> str:
        """Get storage path for a file"""
        return f"{job_id}/{filename}"
        
    def create_job_directory(self, job_id: str) -> None:
        """Supabase Storage creates paths on the fly during upload"""
        pass

    def save_status(self, job_id: str, status_data: Dict[str, Any]) -> None:
        """Save job status to Supabase Storage"""
        path = self._get_path(job_id, "status.json")
        try:
            status_data['last_updated'] = datetime.now().isoformat()
            data_str = json.dumps(status_data, indent=2)
            
            # Upsert the file
            self.supabase.storage.from_(self.bucket_name).upload(
                path=path,
                file=data_str.encode('utf-8'),
                file_options={"cache-control": "3600", "upsert": "true"}
            )
        except Exception as e:
            logger.error(f"Failed to save status to Supabase for job {job_id}: {e}")

    def load_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Load job status from Supabase Storage"""
        path = self._get_path(job_id, "status.json")
        try:
            response = self.supabase.storage.from_(self.bucket_name).download(path)
            return json.loads(response.decode('utf-8'))
        except Exception as e:
            # Check if it's a 404 (file not found)
            if "The resource was not found" in str(e) or "Object not found" in str(e):
                return None
            logger.error(f"Failed to load status from Supabase for job {job_id}: {e}")
            return None

    def append_text(self, job_id: str, text: str, page_num: int) -> None:
        """Append text to result file in Supabase (Download -> Append -> Upload)"""
        path = self._get_path(job_id, "result.txt")
        try:
            current_text = ""
            try:
                response = self.supabase.storage.from_(self.bucket_name).download(path)
                current_text = response.decode('utf-8')
                if current_text:
                    current_text += "\n\n"
            except Exception:
                # File might not exist yet
                pass
            
            new_text = current_text + f"--- Page {page_num} ---\n" + text
            self.supabase.storage.from_(self.bucket_name).upload(
                path=path,
                file=new_text.encode('utf-8'),
                file_options={"upsert": "true"}
            )
        except Exception as e:
            logger.error(f"Failed to append text to Supabase for job {job_id}: {e}")

    # ------------------------------------------------------------------
    # Structured page storage (mirrors FileHandler)
    # ------------------------------------------------------------------

    def save_page_structure(self, job_id: str, page_num: int, page_dict: dict) -> None:
        """Save structured page result as JSON to Supabase."""
        path = self._get_path(job_id, f"pages/page_{page_num:04d}.json")
        try:
            data_str = json.dumps(page_dict, indent=2, ensure_ascii=False)
            self.supabase.storage.from_(self.bucket_name).upload(
                path=path,
                file=data_str.encode('utf-8'),
                file_options={"upsert": "true"}
            )
        except Exception as e:
            logger.error(f"Failed to save page structure to Supabase for job {job_id}, page {page_num}: {e}")

    def load_page_structure(self, job_id: str, page_num: int) -> Optional[dict]:
        """Load a single page's structured JSON from Supabase."""
        path = self._get_path(job_id, f"pages/page_{page_num:04d}.json")
        try:
            response = self.supabase.storage.from_(self.bucket_name).download(path)
            return json.loads(response.decode('utf-8'))
        except Exception:
            return None

    def build_combined_structured_result(self, job_id: str) -> Optional[dict]:
        """Combine all per-page JSONs into one result.json in Supabase."""
        try:
            files = self.supabase.storage.from_(self.bucket_name).list(f"{job_id}/pages")
            if not files:
                return None

            page_files = sorted(
                [f['name'] for f in files if f['name'].startswith('page_') and f['name'].endswith('.json')]
            )
            if not page_files:
                return None

            pages = []
            for pf in page_files:
                path = self._get_path(job_id, f"pages/{pf}")
                try:
                    response = self.supabase.storage.from_(self.bucket_name).download(path)
                    pages.append(json.loads(response.decode('utf-8')))
                except Exception:
                    continue

            combined = {"total_pages": len(pages), "pages": pages}
            out_path = self._get_path(job_id, "result.json")
            data_str = json.dumps(combined, indent=2, ensure_ascii=False)
            self.supabase.storage.from_(self.bucket_name).upload(
                path=out_path,
                file=data_str.encode('utf-8'),
                file_options={"upsert": "true"}
            )
            logger.info(f"Built combined result.json in Supabase for job {job_id} ({len(pages)} pages)")
            return combined
        except Exception as e:
            logger.error(f"Failed to build combined result in Supabase for job {job_id}: {e}")
            return None

    def read_structured_result(self, job_id: str) -> Optional[dict]:
        """Read the combined result.json from Supabase."""
        path = self._get_path(job_id, "result.json")
        try:
            response = self.supabase.storage.from_(self.bucket_name).download(path)
            return json.loads(response.decode('utf-8'))
        except Exception:
            return None

    def read_result(self, job_id: str) -> Optional[str]:
        """Read result text from Supabase Storage"""
        path = self._get_path(job_id, "result.txt")
        try:
            response = self.supabase.storage.from_(self.bucket_name).download(path)
            return response.decode('utf-8')
        except Exception:
            return None

    def job_exists(self, job_id: str) -> bool:
        """Check if job exists in Supabase Storage"""
        path = self._get_path(job_id, "status.json")
        try:
            # There is no direct 'exists' method in some supabase-py versions, 
            # so we try to get public URL or list
            files = self.supabase.storage.from_(self.bucket_name).list(job_id)
            return any(f['name'] == 'status.json' for f in files)
        except Exception:
            return False

    def log_error(self, job_id: str, error_message: str) -> None:
        """Log error to Supabase Storage"""
        path = self._get_path(job_id, "error.log")
        try:
            current_log = ""
            try:
                response = self.supabase.storage.from_(self.bucket_name).download(path)
                current_log = response.decode('utf-8')
            except Exception:
                pass
            
            timestamp = datetime.now().isoformat()
            new_log = current_log + f"[{timestamp}] {error_message}\n"
            self.supabase.storage.from_(self.bucket_name).upload(
                path=path,
                file=new_log.encode('utf-8'),
                file_options={"upsert": "true"}
            )
        except Exception as e:
            logger.error(f"Failed to log error to Supabase for job {job_id}: {e}")

    def upload_pdf(self, job_id: str, local_path: Path) -> str:
        """Upload source PDF to Supabase persistence"""
        path = self._get_path(job_id, "source.pdf")
        try:
            with open(local_path, "rb") as f:
                self.supabase.storage.from_(self.bucket_name).upload(
                    path=path,
                    file=f,
                    file_options={"upsert": "true"}
                )
            return path
        except Exception as e:
            logger.error(f"Failed to upload PDF to Supabase: {e}")
            raise

    def download_source_pdf(self, job_id: str, target_path: Path) -> bool:
        """Download source PDF from Supabase to local path"""
        path = self._get_path(job_id, "source.pdf")
        try:
            response = self.supabase.storage.from_(self.bucket_name).download(path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_path, "wb") as f:
                f.write(response)
            logger.info(f"Downloaded source PDF from Supabase for job {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to download source PDF from Supabase for job {job_id}: {e}")
            return False

    def list_jobs(self) -> List[str]:
        """List all job IDs in Supabase Storage"""
        try:
            # Root list returns prefixes (folders)
            entries = self.supabase.storage.from_(self.bucket_name).list()
            # Job IDs are folder names
            return [e['name'] for e in entries if e.get('id') is None]  # Folders usually have no id in list response
        except Exception as e:
            logger.error(f"Failed to list jobs in Supabase: {e}")
            return []

    def cleanup_job(self, job_id: str) -> None:
        """Delete job 'directory' and all its contents in Supabase"""
        try:
            # List top-level files
            files = self.supabase.storage.from_(self.bucket_name).list(job_id)
            if not files:
                return

            paths_to_delete = [f"{job_id}/{f['name']}" for f in files if f.get('id') is not None]

            # Also list and delete pages/ sub-folder contents
            try:
                page_files = self.supabase.storage.from_(self.bucket_name).list(f"{job_id}/pages")
                if page_files:
                    paths_to_delete += [f"{job_id}/pages/{f['name']}" for f in page_files]
            except Exception:
                pass

            if paths_to_delete:
                self.supabase.storage.from_(self.bucket_name).remove(paths_to_delete)
            logger.info(f"Cleaned up Supabase job storage: {job_id}")
        except Exception as e:
            logger.error(f"Failed to cleanup Supabase job {job_id}: {e}")
