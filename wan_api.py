"""DashScope WAN 2.5 Video Generation API Client"""
import time
import os
from typing import Optional, Dict, Any, Callable
from http import HTTPStatus
import dashscope


class DashScopeClient:
    """Client for Alibaba Cloud DashScope WAN 2.5 Video Generation API"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize DashScope client
        
        Args:
            api_key: DashScope API key (if not provided, uses DASHSCOPE_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required: pass api_key or set DASHSCOPE_API_KEY environment variable"
            )
        dashscope.api_key = self.api_key

    def submit_generation(
        self,
        prompt: str,
        image_url: Optional[str] = None,
        duration: int = 5,
        width: int = 1280,
        height: int = 720,
        fps: int = 24,
        seed: Optional[int] = None,
    ) -> Optional[str]:
        """
        Submit video generation task (async_call pattern)
        
        Args:
            prompt: Text description for video generation
            image_url: Optional first frame image URL (img2video mode)
            duration: Video duration in seconds (default: 5)
            width: Video width in pixels (default: 1280)
            height: Video height in pixels (default: 720)
            fps: Frames per second (default: 24)
            seed: Random seed for reproducibility
        
        Returns:
            Task ID on success, None on failure
        """
        try:
            input_params = {"prompt": prompt}
            if image_url:
                input_params["image_url"] = image_url

            parameters = {
                "duration": duration,
                "size": f"{width}x{height}",
                "fps": fps,
            }
            if seed is not None:
                parameters["seed"] = seed

            response = dashscope.VideoSynthesis.async_call(
                model="wan25-turbo",
                input=input_params,
                parameters=parameters,
            )

            if response.status_code == HTTPStatus.OK:
                task_id = response.output.get("task_id")
                if task_id:
                    return task_id
                else:
                    print(f"No task_id in response: {response}")
                    return None
            else:
                print(
                    f"Error submitting generation: {response.status_code} - {response.message}"
                )
                return None
        except Exception as e:
            print(f"Exception during generation submission: {e}")
            return None

    def check_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Check task status (fetch pattern)
        
        Args:
            task_id: Task ID
        
        Returns:
            Status info dict with keys: status, progress, video_url (if completed)
            None on error
        """
        try:
            response = dashscope.VideoSynthesis.fetch(task_id=task_id)

            if response.status_code == HTTPStatus.OK:
                output = response.output
                status = output.get("task_status", "UNKNOWN")
                
                result = {
                    "status": self._normalize_status(status),
                    "progress": self._calculate_progress(status),
                    "task_id": task_id,
                }
                
                if status == "SUCCEEDED":
                    video_url = output.get("video_url")
                    if video_url:
                        result["video_url"] = video_url
                elif status == "FAILED":
                    result["error"] = output.get("message", "Unknown error")
                
                return result
            else:
                print(f"Error checking status: {response.status_code} - {response.message}")
                return None
        except Exception as e:
            print(f"Exception checking status: {e}")
            return None

    def _normalize_status(self, dashscope_status: str) -> str:
        """
        Normalize DashScope status to standard format
        
        DashScope statuses: PENDING, RUNNING, SUCCEEDED, FAILED
        """
        status_map = {
            "PENDING": "pending",
            "RUNNING": "processing",
            "SUCCEEDED": "completed",
            "FAILED": "failed",
        }
        return status_map.get(dashscope_status, "unknown")

    def _calculate_progress(self, status: str) -> int:
        """
        Estimate progress percentage based on status
        """
        progress_map = {
            "PENDING": 10,
            "RUNNING": 50,
            "SUCCEEDED": 100,
            "FAILED": 0,
        }
        return progress_map.get(status, 0)

    def wait_for_completion(
        self,
        task_id: str,
        max_wait_time: int = 1200,
        poll_interval: int = 5,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Optional[str]:
        """
        Wait for task completion and return video URL (wait pattern)
        
        Args:
            task_id: Task ID
            max_wait_time: Maximum wait time in seconds (default: 1200)
            poll_interval: Status check interval in seconds (default: 5)
            progress_callback: Optional callback function for progress updates
        
        Returns:
            Video URL on success, None on failure/timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            status_info = self.check_status(task_id)
            
            if status_info is None:
                time.sleep(poll_interval)
                continue
            
            status = status_info.get("status")
            progress = status_info.get("progress", 0)
            
            # Update progress callback
            if progress_callback:
                try:
                    progress_callback(float(progress) / 100.0)
                except Exception:
                    pass
            
            if status == "completed":
                video_url = status_info.get("video_url")
                if video_url:
                    return video_url
                else:
                    print("Task completed but no video_url found")
                    return None
            elif status == "failed":
                error = status_info.get("error", "Unknown error")
                print(f"Generation failed: {error}")
                return None
            elif status in ["pending", "processing"]:
                # Continue waiting
                time.sleep(poll_interval)
            else:
                print(f"Unknown status: {status}")
                time.sleep(poll_interval)
        
        print(f"Timeout: exceeded {max_wait_time} seconds")
        return None


if __name__ == "__main__":
    # Example usage
    API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
    
    if not API_KEY:
        print("❌ Set DASHSCOPE_API_KEY environment variable")
        raise SystemExit(1)
    
    print("Testing DashScope WAN 2.5 video generation...")
    
    try:
        client = DashScopeClient(API_KEY)
        
        print("\nSubmitting test video generation task...")
        task_id = client.submit_generation(
            prompt="A beautiful sunset over the ocean with birds flying in the sky",
            duration=5,
            width=1280,
            height=720,
            fps=24,
        )
        
        if task_id:
            print(f"Task created: {task_id}")
            print("Waiting for generation to complete...")
            
            video_url = client.wait_for_completion(task_id)
            
            if video_url:
                print(f"✅ Video generated successfully: {video_url}")
            else:
                print("❌ Video generation failed")
        else:
            print("❌ Failed to create task")
    except Exception as e:
        print(f"❌ Error: {e}")
