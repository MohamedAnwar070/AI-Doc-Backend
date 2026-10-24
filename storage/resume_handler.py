from typing import Optional, Set, Any
import logging

logger = logging.getLogger(__name__)


class ResumeHandler:
    """Handle job resumption after crashes"""
    
    def __init__(self, file_handler: Any):
        self.file_handler = file_handler
    
    def get_processed_pages(self, job_id: str) -> Set[int]:
        """
        Get set of already processed page numbers
        
        Args:
            job_id: Job identifier
            
        Returns:
            Set of processed page numbers
        """
        status = self.file_handler.load_status(job_id)
        
        if not status:
            return set()
        
        processed = status.get('processed_pages', [])
        return set(processed)
    
    def should_resume(self, job_id: str) -> bool:
        """
        Check if job should be resumed
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if job should be resumed
        """
        status = self.file_handler.load_status(job_id)
        
        if not status:
            return False
        
        current_status = status.get('status')

        # Resume if status is running, failed, pending, or cancelled
        if current_status in ['running', 'failed', 'pending', 'cancelled']:
            return True

        # Allow resume of "completed" jobs that have unprocessed/failed pages
        if current_status == 'completed':
            failed = status.get('failed_pages', [])
            return len(failed) > 0

        return False
    
    def get_next_page_to_process(
        self,
        job_id: str,
        total_pages: int
    ) -> Optional[int]:
        """
        Get next page number that needs processing
        
        Args:
            job_id: Job identifier
            total_pages: Total number of pages
            
        Returns:
            Next page number or None if all pages processed
        """
        processed = self.get_processed_pages(job_id)
        
        # Find first unprocessed page
        for page_num in range(1, total_pages + 1):
            if page_num not in processed:
                return page_num
        
        return None
    
    def get_resume_info(self, job_id: str) -> dict:
        """
        Get information about resume state
        
        Args:
            job_id: Job identifier
            
        Returns:
            Dictionary with resume information
        """
        status = self.file_handler.load_status(job_id)
        
        if not status:
            return {
                'can_resume': False,
                'reason': 'No status file found'
            }
        
        processed_pages = set(status.get('processed_pages', []))
        total_pages = status.get('total_pages', 0)
        
        return {
            'can_resume': self.should_resume(job_id),
            'status': status.get('status'),
            'total_pages': total_pages,
            'processed_count': len(processed_pages),
            'remaining_count': total_pages - len(processed_pages),
            'percentage_complete': (len(processed_pages) / total_pages * 100) if total_pages > 0 else 0,
            'processed_pages': sorted(list(processed_pages))
        }
    
    def mark_for_resume(self, job_id: str) -> None:
        """
        Mark a job to be resumed (change status from failed/running to pending)
        
        Args:
            job_id: Job identifier
        """
        status = self.file_handler.load_status(job_id)
        
        if status:
            status['status'] = 'pending'
            self.file_handler.save_status(job_id, status)
            logger.info(f"Job {job_id} marked for resume")
