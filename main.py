"""WAN 2.5 Video Generator with DashScope SDK and Gradio UI"""
import gradio as gr
import os
from wan_api import DashScopeClient


def generate_video(api_key, prompt, image_url, duration, resolution, fps, seed, progress=gr.Progress()):
    """
    Generate video using DashScope SDK (Alibaba WAN 2.5)
    
    Args:
        api_key: DashScope API key (format: sk-...)
        prompt: Text description for video generation
        image_url: Optional first frame image URL (img2video mode)
        duration: Video duration in seconds
        resolution: Video resolution (format: "1920x1080")
        fps: Frames per second
        seed: Seed for reproducibility
        
    Returns:
        Video URL or error message
    """
    try:
        progress(0, desc="Initializing...")
        
        # Validate API key
        if not api_key or not api_key.strip():
            return None, "❌ Error: Please provide a DashScope API key"
        
        # Create DashScope client
        client = DashScopeClient(api_key=api_key.strip())
        
        progress(0.1, desc="Submitting generation request...")
        
        # Parse resolution
        width, height = map(int, resolution.split('x'))
        
        # Submit video generation task
        task_id = client.submit_generation(
            prompt=prompt,
            image_url=image_url if image_url and image_url.strip() else None,
            duration=duration,
            width=width,
            height=height,
            fps=fps,
            seed=seed if seed > 0 else None
        )
        
        if not task_id:
            return None, "❌ Error: Failed to submit generation request"
        
        progress(0.2, desc=f"Video generation started (ID: {task_id})...")
        
        # Wait for completion with progress updates
        video_url = client.wait_for_completion(
            task_id, 
            progress_callback=lambda p: progress(0.2 + p * 0.7, desc="Generating video...")
        )
        
        if video_url:
            progress(1.0, desc="Done!")
            return video_url, f"✅ Video generated successfully!\nTask ID: {task_id}\nVideo URL: {video_url}"
        else:
            return None, "❌ Error: Video generation failed"
            
    except ValueError as e:
        return None, f"❌ Error: {str(e)}"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def create_interface():
    """
    Create Gradio interface for WAN 2.5 video generation
    """
    with gr.Blocks(title="WAN 2.5 Video Generator (DashScope)", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎬 WAN 2.5 Video Generator (DashScope SDK)
            
            Generate videos using the official DashScope SDK for Alibaba WAN 2.5.
            
            ## How to use:
            1. Get your DashScope API key (format: sk-...)
            2. Enter your API key below
            3. Provide a text description (and optionally an image URL)
            4. Configure generation parameters
            5. Click "Generate Video"
            
            **Note:** The generated video URL will be provided directly from DashScope API.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🔐 Authentication")
                
                api_key = gr.Textbox(
                    label="DashScope API Key",
                    placeholder="sk-************************",
                    type="password",
                    info="Key is used locally and sent to DashScope API only"
                )
                
                gr.Markdown("### 📝 Generation Parameters")
                
                prompt = gr.Textbox(
                    label="Video Description (Prompt)",
                    placeholder="A serene sunset over the ocean with birds flying...",
                    lines=3,
                    info="Describe what you want to see in the video"
                )
                
                image_url = gr.Textbox(
                    label="Image URL (Optional - for img2video)",
                    placeholder="https://example.com/image.jpg",
                    info="Provide an image URL to use as the first frame (img2video mode)"
                )
                
                with gr.Row():
                    duration = gr.Slider(
                        label="Duration (sec)",
                        minimum=1,
                        maximum=30,
                        value=5,
                        step=1
                    )
                    fps = gr.Slider(
                        label="FPS",
                        minimum=8,
                        maximum=60,
                        value=24,
                        step=1
                    )
                
                resolution = gr.Dropdown(
                    label="Resolution",
                    choices=["512x512", "768x768", "1024x576", "1280x720", "1920x1080"],
                    value="1280x720"
                )
                
                seed = gr.Number(
                    label="Seed (optional)",
                    value=-1,
                    precision=0,
                    info="Use -1 for random seed"
                )
                
                generate_btn = gr.Button("🎬 Generate Video", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                gr.Markdown("### 🎬 Result")
                
                output_video = gr.Video(label="Generated Video")
                output_status = gr.Textbox(label="Status", lines=4)
                
                gr.Markdown(
                    """
                    ### 💡 Tips:
                    - Use detailed descriptions for best results
                    - Higher resolutions require more time and resources
                    - Seed allows reproducing identical results
                    - DashScope API key starts with "sk-"
                    - Image URL enables img2video mode (first frame)
                    """
                )
        
        # Bind generation function to button
        generate_btn.click(
            fn=generate_video,
            inputs=[api_key, prompt, image_url, duration, resolution, fps, seed],
            outputs=[output_video, output_status]
        )
        
        gr.Markdown(
            """
            ---
            ### 📚 Documentation
            - Uses official DashScope SDK (Python)
            - API Pattern: `async_call` → `fetch` → `wait`
            - Ensure API key is valid and has access to WAN 2.5
            - Check logs and API limits if issues occur
            - Documentation: https://help.aliyun.com/zh/dashscope/
            """
        )
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",  # Access from local network
        server_port=7860,
        share=False,  # Set True for public access via Gradio
        show_error=True
    )
