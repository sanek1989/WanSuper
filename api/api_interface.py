"""WAN 2.5 Video Generator API Interface with Gradio UI"""
import gradio as gr
import os
from .wan_api import DashScopeClient

def generate_video(api_key, mode, prompt, image_file, image_url, duration, resolution, fps, seed, progress=gr.Progress()):
    """
    Generate video using DashScope SDK (Alibaba WAN 2.5)
    
    Args:
        api_key: DashScope API key (format: sk-...)
        mode: Generation mode ("text2video" or "img2video")
        prompt: Text description for video generation
        image_file: Local image file (for img2video mode)
        image_url: Image URL (for img2video mode)
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
        
        # Determine image source based on mode
        image_input = None
        if mode == "img2video":
            if image_file is not None:
                # Local file upload (priority)
                image_input = image_file
                progress(0.15, desc="Processing local image...")
            elif image_url and image_url.strip():
                # URL input
                image_input = image_url.strip()
            else:
                return None, "❌ Error: img2video mode requires an image (upload or URL)"
        
        # Submit video generation task
        task_id = client.submit_generation(
            prompt=prompt,
            image_url=image_input,
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
            mode_desc = "img2video" if mode == "img2video" else "text2video"
            return video_url, f"✅ Video generated successfully!\nMode: {mode_desc}\nTask ID: {task_id}\nVideo URL: {video_url}"
        else:
            return None, "❌ Error: Video generation failed"
            
    except ValueError as e:
        return None, f"❌ Error: {str(e)}"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

def update_image_inputs(mode):
    """
    Update visibility of image inputs based on selected mode
    """
    if mode == "img2video":
        return [
            gr.update(visible=True),  # image_file
            gr.update(visible=True)   # image_url
        ]
    else:
        return [
            gr.update(visible=False),
            gr.update(visible=False)
        ]

def create_api_interface():
    """
    Create Gradio interface for WAN 2.5 API mode video generation
    """
    with gr.Blocks(title="WAN 2.5 API Mode - The Angel Studio", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎞️ WAN 2.5 API Mode - Cloud Generation
            
            Generate videos using Alibaba Cloud DashScope WAN 2.5 API.

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
                
                gr.Markdown("### 🎥 Generation Mode")
                
                mode = gr.Radio(
                    label="Mode",
                    choices=["text2video", "img2video"],
                    value="text2video",
                    info="Select text2video for pure text generation, or img2video to use an image as the first frame"
                )
                
                gr.Markdown("### 📝 Generation Parameters")
                
                prompt = gr.Textbox(
                    label="Video Description (Prompt)",
                    placeholder="A serene sunset over the ocean with birds flying...",
                    lines=3,
                    info="Describe what you want to see in the video"
                )
                
                # Image inputs (hidden by default for text2video)
                image_file = gr.Image(
                    label="Upload Image (for img2video)",
                    type="filepath",
                    visible=False
                )
                
                image_url = gr.Textbox(
                    label="OR Image URL (for img2video)",
                    placeholder="https://example.com/image.jpg",
                    visible=False,
                    info="Alternatively, provide an image URL (used if no file uploaded)"
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
                output_status = gr.Textbox(label="Status", lines=5)
                
                gr.Markdown(
                    """
                    ### 💡 Tips:
                    - **text2video**: Generate video from text description only
                    - **img2video**: Use an image as the first frame and animate it
                    - Use detailed descriptions for best results
                    - Higher resolutions require more time and resources
                    - Seed allows reproducing identical results
                    - DashScope API key starts with "sk-"
                    - Local image upload has priority over URL
                    """
                )
        
        # Mode change handler - show/hide image inputs
        mode.change(
            fn=update_image_inputs,
            inputs=[mode],
            outputs=[image_file, image_url]
        )
        
        # Bind generation function to button
        generate_btn.click(
            fn=generate_video,
            inputs=[api_key, mode, prompt, image_file, image_url, duration, resolution, fps, seed],
            outputs=[output_video, output_status]
        )
        
        gr.Markdown(
            """
            ---
            **Made with ❤️ by The Angel Studio**
            """
        )
    
    return demo
