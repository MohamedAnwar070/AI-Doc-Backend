#!/usr/bin/env python3
"""
Example usage demonstrations for PDF OCR Backend
"""
import requests
import time
from pathlib import Path


def example_1_simple_upload():
    """Example 1: Simple PDF upload and wait for completion"""
    print("\n" + "="*60)
    print("Example 1: Simple Upload and Wait")
    print("="*60)
    
    # Upload PDF
    print("\n1. Uploading PDF...")
    with open('sample.pdf', 'rb') as f:
        response = requests.post(
            'http://localhost:8001/api/upload',
            files={'file': ('sample.pdf', f, 'application/pdf')}
        )
    
    job_data = response.json()
    job_id = job_data['job_id']
    print(f"   Job ID: {job_id}")
    print(f"   Total pages: {job_data['total_pages']}")
    
    # Poll status until complete
    print("\n2. Waiting for completion...")
    while True:
        status = requests.get(f'http://localhost:8001/api/status/{job_id}').json()
        
        if status['status'] in ['completed', 'failed']:
            break
        
        print(f"   Progress: {status['processed_pages']}/{status['total_pages']} "
              f"({status['percentage']:.1f}%)")
        time.sleep(5)
    
    # Get result
    print("\n3. Fetching result...")
    result = requests.get(f'http://localhost:8001/api/result/{job_id}').json()
    print(f"   Status: {result['status']}")
    print(f"   Extracted {len(result['text'])} characters")
    
    print(f"\n✓ Complete! Job ID: {job_id}")


def example_2_async_upload():
    """Example 2: Upload and don't wait (check later)"""
    print("\n" + "="*60)
    print("Example 2: Async Upload (Upload and Check Later)")
    print("="*60)
    
    # Upload multiple PDFs
    pdf_files = ['doc1.pdf', 'doc2.pdf', 'doc3.pdf']
    job_ids = []
    
    print("\n1. Uploading multiple PDFs...")
    for pdf_file in pdf_files:
        with open(pdf_file, 'rb') as f:
            response = requests.post(
                'http://localhost:8001/api/upload',
                files={'file': (pdf_file, f, 'application/pdf')}
            )
        
        job_data = response.json()
        job_ids.append(job_data['job_id'])
        print(f"   Uploaded {pdf_file} → Job {job_data['job_id']}")
    
    print(f"\n2. All uploads complete. Jobs: {len(job_ids)}")
    print("   You can now close this script and check status later!")
    
    # Show how to check status later
    print("\n3. To check status later, use:")
    for job_id in job_ids:
        print(f"   curl http://localhost:8001/api/status/{job_id}")


def example_3_download_result():
    """Example 3: Download result as file"""
    print("\n" + "="*60)
    print("Example 3: Download Result File")
    print("="*60)
    
    job_id = input("\nEnter job ID: ").strip()
    
    # Check if job exists
    status_response = requests.get(f'http://localhost:8001/api/status/{job_id}')
    
    if status_response.status_code != 200:
        print(f"✗ Job {job_id} not found")
        return
    
    status = status_response.json()
    print(f"\nJob Status: {status['status']}")
    print(f"Progress: {status['processed_pages']}/{status['total_pages']}")
    
    if status['status'] != 'completed':
        print("\n⚠ Job not yet completed. Wait for completion first.")
        return
    
    # Download result
    print("\nDownloading result...")
    download_response = requests.get(
        f'http://localhost:8001/api/result/{job_id}/download'
    )
    
    output_file = f'result_{job_id}.txt'
    with open(output_file, 'wb') as f:
        f.write(download_response.content)
    
    print(f"✓ Downloaded to: {output_file}")


def example_4_error_handling():
    """Example 4: Proper error handling"""
    print("\n" + "="*60)
    print("Example 4: Error Handling")
    print("="*60)
    
    try:
        # Try to upload non-PDF file
        print("\n1. Attempting to upload non-PDF file...")
        with open('document.txt', 'wb') as f:
            f.write(b'This is not a PDF')
        
        with open('document.txt', 'rb') as f:
            response = requests.post(
                'http://localhost:8001/api/upload',
                files={'file': ('document.txt', f)}
            )
        
        if response.status_code != 200:
            error_data = response.json()
            print(f"   ✗ Upload rejected: {error_data.get('detail')}")
    
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Try to get status of non-existent job
    print("\n2. Checking non-existent job...")
    response = requests.get('http://localhost:8001/api/status/fake-job-id')
    
    if response.status_code == 404:
        print(f"   ✓ Properly returned 404: {response.json().get('detail')}")


def example_5_monitoring_multiple_jobs():
    """Example 5: Monitor multiple jobs"""
    print("\n" + "="*60)
    print("Example 5: Monitor Multiple Jobs")
    print("="*60)
    
    # Get all job directories
    import os
    output_dir = 'outputs'
    
    if not os.path.exists(output_dir):
        print("No jobs found")
        return
    
    job_ids = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    
    print(f"\nFound {len(job_ids)} jobs:")
    print("-" * 60)
    
    for job_id in job_ids:
        response = requests.get(f'http://localhost:8001/api/status/{job_id}')
        
        if response.status_code == 200:
            status = response.json()
            print(f"\nJob: {job_id[:8]}...")
            print(f"  File: {status.get('pdf_filename', 'N/A')}")
            print(f"  Status: {status['status']}")
            print(f"  Progress: {status['processed_pages']}/{status['total_pages']}")
            print(f"  Engine: {status.get('engine', 'N/A')}")


def example_6_cancel_delete_job():
    """Example 6: Cancel and delete jobs"""
    print("\n" + "="*60)
    print("Example 6: Cancel and Delete Jobs")
    print("="*60)
    
    job_id = input("\nEnter job ID to cancel: ").strip()
    
    # Cancel job
    print("\n1. Cancelling job...")
    response = requests.post(f'http://localhost:8001/api/cancel/{job_id}')
    
    if response.status_code == 200:
        print(f"   ✓ Job cancelled: {response.json().get('message')}")
    else:
        print(f"   ✗ Failed to cancel: {response.json().get('detail')}")
        return
    
    # Delete job
    confirm = input("\n2. Delete job files? (yes/no): ").strip().lower()
    
    if confirm == 'yes':
        response = requests.delete(f'http://localhost:8001/api/delete/{job_id}')
        
        if response.status_code == 200:
            print(f"   ✓ Job deleted: {response.json().get('message')}")
        else:
            print(f"   ✗ Failed to delete: {response.json().get('detail')}")


def main():
    """Main menu"""
    print("\n" + "="*60)
    print("PDF OCR Backend - Example Usage")
    print("="*60)
    
    examples = {
        '1': ('Simple Upload and Wait', example_1_simple_upload),
        '2': ('Async Upload (Multiple Files)', example_2_async_upload),
        '3': ('Download Result File', example_3_download_result),
        '4': ('Error Handling Demo', example_4_error_handling),
        '5': ('Monitor Multiple Jobs', example_5_monitoring_multiple_jobs),
        '6': ('Cancel and Delete Job', example_6_cancel_delete_job),
    }
    
    print("\nAvailable Examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    
    print("\n  0. Exit")
    
    while True:
        choice = input("\nSelect example (0-6): ").strip()
        
        if choice == '0':
            print("\nGoodbye!")
            break
        
        if choice in examples:
            name, func = examples[choice]
            try:
                func()
            except Exception as e:
                print(f"\n✗ Error running example: {e}")
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()
