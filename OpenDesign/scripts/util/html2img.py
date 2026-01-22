import os
from playwright.sync_api import sync_playwright
from PIL import Image
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import time


def take_screenshot(html_content, output_file="tmp_screenshot.png", output_dir="tmp", timeout=60):
    """
    Args:
        html_content: str, the html content to be screenshot
        output_file: str, the name of the screenshot
        output_dir: str, the directory to save the screenshot
        timeout: int, the timeout in seconds
    """
    # Save HTML content to temporary file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    temp_html_file = os.path.join(output_dir, f"{output_file}_tmp.html")
    with open(temp_html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    url = temp_html_file
    # Convert local path to file:// URL if it's a file
    if os.path.exists(url):
        url = "file://" + os.path.abspath(url)

    try:
        with sync_playwright() as p:
            # Choose a browser, e.g., Chromium, Firefox, or WebKit
            browser = p.chromium.launch()
            page = browser.new_page()

            # Navigate to the URL
            page.goto(url, timeout=timeout*1000)

            # Take the screenshot
            page.screenshot(path=os.path.join(output_dir, output_file), full_page=True, animations="disabled", timeout=timeout*1000)

            browser.close()
    except Exception as e: 
        print(f"Failed to take screenshot due to: {e}. Generating a blank image.")
        # Generate a blank image 
        img = Image.new('RGB', (1280, 960), color = 'white')
        img.save(output_file)
    
    # Clean up temporary HTML file
    if os.path.exists(temp_html_file):
        os.remove(temp_html_file)


def take_screenshot_single_file(html_file_path, output_path, timeout=60):
    """
    Take screenshot of a single HTML file.
    
    Args:
        html_file_path: str, path to the HTML file
        output_path: str, path to save the screenshot
        timeout: int, timeout in seconds
    
    Returns:
        tuple: (success: bool, file_path: str, error_msg: str or None)
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            
            url = "file://" + os.path.abspath(html_file_path)
            page.goto(url, timeout=timeout*1000)
            page.screenshot(path=output_path, full_page=True, animations="disabled", timeout=timeout*1000)
            
            browser.close()
            return True, html_file_path, None
    except Exception as e:
        # Generate a blank image on error
        try:
            img = Image.new('RGB', (1280, 960), color='white')
            img.save(output_path)
            return False, html_file_path, str(e)
        except Exception as save_error:
            return False, html_file_path, f"Screenshot failed: {e}, Blank image save failed: {save_error}"


def take_screenshot_dir(html_dir, max_workers=4, timeout=60):
    """
    Take screenshots of all HTML files in a directory using multithreading.
    
    Args:
        html_dir: str, the directory containing HTML files
        max_workers: int, maximum number of worker threads
        timeout: int, timeout in seconds for each screenshot
    """
    html_files = [f for f in os.listdir(html_dir) if f.endswith(".html")]
    
    if not html_files:
        print("No HTML files found in the directory.")
        return
    
    print(f"Found {len(html_files)} HTML files. Processing with {max_workers} threads...")
    
    # Prepare tasks
    tasks = []
    for file in html_files:
        file_path = os.path.join(html_dir, file)
        output_path = file_path.replace(".html", ".png")
        tasks.append((file_path, output_path, timeout))
    
    # Process files with multithreading
    success_count = 0
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(take_screenshot_single_file, file_path, output_path, timeout): file_path
            for file_path, output_path, timeout in tasks
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(tasks), desc="Taking screenshots") as pbar:
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    success, processed_file, error_msg = future.result()
                    if success:
                        success_count += 1
                    else:
                        error_count += 1
                        file_name = os.path.basename(processed_file)
                        print(f"Failed to take screenshot for {file_name}: {error_msg}. Generated blank image.")
                except Exception as e:
                    error_count += 1
                    file_name = os.path.basename(file_path)
                    print(f"Unexpected error processing {file_name}: {e}")
                
                pbar.update(1)
    
    print(f"Screenshot processing completed. Success: {success_count}, Errors: {error_count}")
    
    # Clean up all HTML files in the directory
    print("Cleaning up temporary HTML files...")
    cleanup_count = 0
    for file in html_files:
        file_path = os.path.join(html_dir, file)
        if os.path.exists(file_path):
            try:    
                os.remove(file_path)
                cleanup_count += 1
            except Exception as e:
                print(f"Failed to remove temporal html file {file_path} due to: {e}")
    
    print(f"Cleaned up {cleanup_count} HTML files.")


def take_screenshot_dir_optimized(html_dir, max_workers=4, timeout=120, batch_size=50):
    """
    Optimized version with resource management and batch processing.
    Take screenshots of all HTML files in a directory using multithreading with periodic browser restart.
    
    Args:
        html_dir: str, the directory containing HTML files
        max_workers: int, maximum number of worker threads (reduced for stability)
        timeout: int, timeout in seconds for each screenshot
        batch_size: int, number of files to process before restarting browsers
    """
    html_files = [f for f in os.listdir(html_dir) if f.endswith(".html")]
    
    if not html_files:
        print("No HTML files found in the directory.")
        return
    
    print(f"Found {len(html_files)} HTML files. Processing with {max_workers} threads in batches of {batch_size}...")
    
    # Thread-local storage for browser instances with usage counter
    thread_local = threading.local()
    
    def get_browser_and_page():
        """Get or create browser and page for current thread with resource limits."""
        # Initialize or restart browser if it has processed too many files
        if (not hasattr(thread_local, 'playwright') or 
            not hasattr(thread_local, 'usage_count') or 
            thread_local.usage_count >= batch_size // max_workers):
            
            # Clean up existing resources first
            if hasattr(thread_local, 'browser'):
                try:
                    thread_local.browser.close()
                except:
                    pass
            if hasattr(thread_local, 'playwright'):
                try:
                    thread_local.playwright.stop()
                except:
                    pass
            
            # Create new browser instance with balanced settings for stability vs functionality
            thread_local.playwright = sync_playwright().start()
            thread_local.browser = thread_local.playwright.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-dev-shm-usage',  # Keep this for Docker/Linux stability
                    '--disable-extensions',     # Safe to disable
                    '--disable-plugins',        # Safe to disable  
                    '--disable-web-security',   # Helps with local file access
                    '--max_old_space_size=4096', # Memory limit to prevent crashes
                    '--no-first-run',           # Skip first run tasks
                    '--disable-default-apps',   # Reduce startup overhead
                ]
            )
            thread_local.page = thread_local.browser.new_page()
            
            # Set reasonable timeout and viewport
            thread_local.page.set_default_timeout(timeout * 1000)
            thread_local.page.set_viewport_size({"width": 1280, "height": 960})
            
            thread_local.usage_count = 0
            
        thread_local.usage_count += 1
        return thread_local.browser, thread_local.page
    
    def cleanup_thread_resources():
        """Clean up browser resources for current thread."""
        if hasattr(thread_local, 'browser'):
            try:
                thread_local.browser.close()
            except:
                pass
        if hasattr(thread_local, 'playwright'):
            try:
                thread_local.playwright.stop()
            except:
                pass
        # Reset thread local
        if hasattr(thread_local, 'usage_count'):
            delattr(thread_local, 'usage_count')
    
    def process_file(file_info):
        """Process a single HTML file with timeout protection."""
        file_path, output_path = file_info
        retry_count = 2  # Reduce retries to speed up
        
        for attempt in range(retry_count):
            try:
                browser, page = get_browser_and_page()
                
                url = "file://" + os.path.abspath(file_path)
                file_name = os.path.basename(file_path)
                
                # Use load wait strategy with timeout protection
                page.goto(url, wait_until="domcontentloaded", timeout=30000)  # Reduce timeout
                
                # Wait for network idle with very short timeout to prevent hanging
                try:
                    page.wait_for_load_state("networkidle", timeout=3000)  # Only 3 seconds
                except:
                    pass  # Continue if networkidle timeout
                
                # Take screenshot immediately after loading
                page.screenshot(
                    path=output_path, 
                    full_page=True, 
                    animations="disabled", 
                    timeout=30000  # Reduce screenshot timeout
                )
                
                return True, file_path, None
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {file_name}: {str(e)[:100]}...")
                if attempt < retry_count - 1:
                    # Clean up and retry
                    cleanup_thread_resources()
                    time.sleep(0.5)  # Shorter wait between retries
                    continue
                else:
                    # Final attempt failed, generate blank image
                    try:
                        img = Image.new('RGB', (1280, 960), color='white')
                        img.save(output_path)
                        print(f"Generated blank image for {file_name}")
                        return False, file_path, str(e)
                    except Exception as save_error:
                        return False, file_path, f"Screenshot failed: {e}, Blank image save failed: {save_error}"
    
    # Process files in batches to prevent resource accumulation
    total_files = len(html_files)
    success_count = 0
    error_count = 0
    
    for batch_start in range(0, total_files, batch_size):
        batch_end = min(batch_start + batch_size, total_files)
        batch_files = html_files[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//batch_size + 1}/{(total_files-1)//batch_size + 1} ({len(batch_files)} files)...")
        
        # Prepare tasks for this batch
        tasks = []
        for file in batch_files:
            file_path = os.path.join(html_dir, file)
            output_path = file_path.replace(".html", ".png")
            tasks.append((file_path, output_path))
        
        # Process batch with multithreading and timeout protection
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks for this batch
            futures = [executor.submit(process_file, task) for task in tasks]
            
            # Process completed tasks with progress bar and timeout
            with tqdm(total=len(tasks), desc=f"Batch {batch_start//batch_size + 1} screenshots") as pbar:
                try:
                    for future in as_completed(futures, timeout=120):  # 2 minute timeout for batch
                        try:
                            success, processed_file, error_msg = future.result(timeout=1)  # Quick result timeout
                            if success:
                                success_count += 1
                            else:
                                error_count += 1
                                file_name = os.path.basename(processed_file)
                                print(f"Failed to take screenshot for {file_name}: {error_msg}. Generated blank image.")
                        except TimeoutError:
                            error_count += 1
                            print(f"Timeout waiting for result - continuing...")
                        except Exception as e:
                            error_count += 1
                            print(f"Unexpected error: {e}")
                        
                        pbar.update(1)
                except TimeoutError:
                    print(f"Batch timeout reached - some files may not be processed")
                    # Cancel remaining futures
                    for future in futures:
                        future.cancel()
            
            # Clean up thread resources for this batch
            for _ in range(max_workers):
                executor.submit(cleanup_thread_resources)
        
        # Small delay between batches to allow system cleanup
        if batch_end < total_files:
            time.sleep(2)
            print(f"Batch completed. Progress: {batch_end}/{total_files} files ({success_count} success, {error_count} errors)")
    
    print(f"Screenshot processing completed. Success: {success_count}, Errors: {error_count}")
    
    # Clean up all HTML files in the directory
    # print("Cleaning up temporary HTML files...")
    # cleanup_count = 0
    # for file in html_files:
    #     file_path = os.path.join(html_dir, file)
    #     if os.path.exists(file_path):
    #         try:    
    #             os.remove(file_path)
    #             cleanup_count += 1
    #         except Exception as e:
    #             print(f"Failed to remove temporal html file {file_path} due to: {e}")
    
    # print(f"Cleaned up {cleanup_count} HTML files.")


# Keep the original function as take_screenshot_dir for backward compatibility
# Users can choose between take_screenshot_dir and take_screenshot_dir_optimized

"""
Usage Examples:

# Basic multithreaded version (creates new browser for each file)
take_screenshot_dir(html_dir, max_workers=4, timeout=60)

# Optimized version (reuses browser instances per thread)
take_screenshot_dir_optimized(html_dir, max_workers=4, timeout=60)

Performance Notes:
- take_screenshot_dir: Simpler implementation, creates new browser for each file
- take_screenshot_dir_optimized: Better performance for large batches, reuses browsers per thread
- Recommended max_workers: 2-8 depending on system resources
- Higher max_workers may not always be better due to browser resource limits
"""