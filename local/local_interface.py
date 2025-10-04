"""WAN 2.5 Local Mode Interface - Placeholder"""
import gradio as gr

def create_local_interface():
    """
    Create Gradio interface for WAN 2.5 Local mode (self-hosted)
    
    This is a placeholder interface for local WAN 2.5 deployment.
    """
    with gr.Blocks(title="WAN 2.5 Local Mode - The Angel Studio", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🏘️ WAN 2.5 Local Mode - Self-Hosted Server
            
            ## 🚧 Coming Soon!
            
            This mode will allow you to generate videos using a locally deployed WAN 2.5 server.
            
            ### Requirements:
            - Locally deployed WAN 2.5 model server
            - Powerful GPU (recommended 24GB+ VRAM)
            - Python 3.8+
            - Local server endpoint URL
            
            ### Features:
            - ✅ Full data privacy
            - ✅ No cloud API keys required  
            - ✅ Unlimited local generation
            - ✅ Complete control over the process
            - ✅ No external dependencies
            
            ### How to Deploy WAN 2.5 Locally:
            
            1. **Get the Model**: Download WAN 2.5 model weights
            2. **Setup Server**: Deploy model inference server
            3. **Configure Endpoint**: Set up local API endpoint
            4. **Test Connection**: Verify server is running
            5. **Use This Interface**: Connect to your local server
            
            ---
            
            ## 💬 Current Status
            
            🚧 **This feature is currently under development.**
            
            For now, please use **API Mode** which connects to Alibaba Cloud DashScope WAN 2.5.
            
            ---
            
            ### 📚 Resources:
            
            - [WAN 2.5 Model Information](https://help.aliyun.com/zh/dashscope/)
            - [Model Deployment Guide](https://help.aliyun.com/zh/dashscope/)
            - [Local Inference Setup](https://github.com/)
            
            ---
            
            **Made with ❤️ by The Angel Studio**
            
            🌟 [Support The Angel Studio on Boosty](https://boosty.to/the_angel)
            """
        )
        
        gr.Markdown(
            """
            ### 🔙 Return to Main Menu
            
            To go back to the main menu and select API Mode, please restart the application.
            """
        )
    
    return demo
