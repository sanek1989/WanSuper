"""WAN 2.5 Video Generator - Main Menu

The Angel Studio - Professional AI Video Generation Tools
"""
import gradio as gr
from api.api_interface import create_api_interface
from local.local_interface import create_local_interface

def create_main_menu():
    """
    Create main menu interface with mode selection
    """
    with gr.Blocks(title="The Angel Studio - WAN 2.5 Video Generator", theme=gr.themes.Soft()) as demo:
        # Large title with The Angel Studio branding
        gr.Markdown(
            """
            <div style="text-align: center; padding: 40px 20px;">
                <h1 style="font-size: 3.5em; margin-bottom: 10px; color: #2c3e50;">
                    🎬 THE ANGEL STUDIO
                </h1>
                <h2 style="font-size: 2em; margin-top: 0; margin-bottom: 20px; color: #34495e;">
                    WAN 2.5 Video Generator
                </h2>
                <p style="font-size: 1.2em; color: #7f8c8d;">
                    Professional AI Video Generation Tools
                </p>
            </div>
            """
        )
        
        gr.Markdown(
            """
            ---
            
            ## 🎮 Select Generation Mode
            
            Choose your preferred video generation mode below.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(
                    """
                    ### 🚫 API Mode
                    
                    **Cloud-based video generation**
                    
                    ✅ Uses Alibaba Cloud DashScope WAN 2.5 API  
                    ✅ No local GPU required  
                    ✅ Fast and convenient  
                    ✅ Requires API key (sk-...)  
                    ✅ Supports text2video & img2video  
                    
                    Perfect for quick generation without heavy hardware requirements.
                    """
                )
                
                api_button = gr.Button(
                    "🔑 Start API Mode",
                    variant="primary",
                    size="lg",
                    scale=1
                )
            
            with gr.Column(scale=1):
                gr.Markdown(
                    """
                    ### 🏘️ Local Mode
                    
                    **Self-hosted server generation**
                    
                    ✅ Complete data privacy  
                    ✅ No cloud API keys required  
                    ✅ Unlimited local generation  
                    ✅ Full control over process  
                    🚧 Requires powerful GPU (24GB+ VRAM)  
                    
                    Ideal for users with local WAN 2.5 deployment and privacy needs.
                    """
                )
                
                local_button = gr.Button(
                    "🏘️ Start Local Mode",
                    variant="secondary",
                    size="lg",
                    scale=1
                )
        
        gr.Markdown(
            """
            ---
            
            ### 📚 About WAN 2.5
            
            WAN 2.5 is a powerful video generation model that creates high-quality videos from text descriptions or images.
            
            - **text2video**: Generate videos from text prompts
            - **img2video**: Animate images into videos
            - **High quality**: Support for various resolutions and frame rates
            - **Flexible**: Customizable duration, resolution, FPS, and seed
            
            ---
            
            ### 🌟 Support The Angel Studio
            
            If you find this tool useful, please consider supporting our work:
            
            [💝 Support on Boosty](https://boosty.to/the_angel) | [🐛 GitHub Repository](https://github.com/sanek1989/WanSuper)
            
            ---
            
            <div style="text-align: center; padding: 20px; color: #95a5a6;">
                <p>Made with ❤️ by <strong>The Angel Studio</strong></p>
                <p style="font-size: 0.9em;">Professional AI Tools for Creative Professionals</p>
            </div>
            """
        )
        
        # Create hidden interfaces that will be shown when buttons are clicked
        api_interface = create_api_interface()
        local_interface = create_local_interface()
        
        # Button click handlers - these will navigate to the respective interfaces
        api_button.click(
            fn=lambda: None,
            outputs=None
        ).then(
            fn=lambda: gr.update(visible=False),
            outputs=demo
        ).then(
            fn=lambda: api_interface.launch(prevent_thread_lock=True),
            outputs=None
        )
        
        local_button.click(
            fn=lambda: None,
            outputs=None
        ).then(
            fn=lambda: gr.update(visible=False),
            outputs=demo
        ).then(
            fn=lambda: local_interface.launch(prevent_thread_lock=True),
            outputs=None
        )
    
    return demo

if __name__ == "__main__":
    print("✨ Starting The Angel Studio - WAN 2.5 Video Generator...")
    print("🌐 Server will be available at: http://localhost:7860")
    print("🌟 Support us: https://boosty.to/the_angel")
    
    # For simplicity, we'll just launch the API interface directly
    # The menu approach with buttons would require more complex state management
    # So we'll create a tabbed interface instead
    
    api_interface = create_api_interface()
    local_interface = create_local_interface()
    
    # Create tabbed interface with The Angel Studio branding
    demo = gr.TabbedInterface(
        [api_interface, local_interface],
        ["🔑 API Mode (Cloud)", "🏘️ Local Mode (Self-Hosted)"],
        title="THE ANGEL STUDIO - WAN 2.5 Video Generator",
        theme=gr.themes.Soft()
    )
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
