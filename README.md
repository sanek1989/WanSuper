# WAN 2.5 Video Generator (DashScope SDK)
🎬 Video generation project using Alibaba Cloud DashScope official SDK for WAN 2.5 with Gradio UI.

## Update: DashScope Official SDK Integration
- Integrated official DashScope Python SDK
- API pattern: `async_call` → `fetch` → `wait`
- Support for both text2video and img2video modes
- **NEW**: Mode selection UI (text2video / img2video)
- **NEW**: Local image file upload for img2video
- Direct video URL from DashScope API
- Updated all documentation to reflect DashScope SDK usage

## 📋 Description
This application provides a simple web interface for working with WAN 2.5 video generation model using the official Alibaba Cloud DashScope SDK.

## ✨ Features
- 🔐 Authentication with DashScope API key (sk-...)
- 🎥 **Mode Selection**: Choose between text2video or img2video generation
- 📝 Text-to-video generation with prompts
- 🖼️ Image-to-video generation (first frame)
  - **Upload local image files** from your PC
  - Or provide an image URL
- ⚙️ Customizable generation parameters:
  - Video duration (1-30 seconds)
  - Resolution (512x512 to 1920x1080)
  - Frames per second (8-60 FPS)
  - Seed for reproducible results
- 📊 Progress tracking
- 🎬 Direct video URL output
- 🎨 User-friendly Gradio interface

## 🚀 Quick Start

### Requirements
- Python 3.8+
- Active DashScope API key with WAN 2.5 access

### Installation
```bash
git clone https://github.com/sanek1989/WanSuper.git
cd WanSuper
pip install -r requirements.txt
```

This will install:
- `dashscope>=1.14.0` - Official DashScope SDK
- `gradio>=4.0.0` - Web UI framework
- `requests>=2.31.0` - HTTP library
- Other dependencies

### Getting DashScope API Key
1. Visit [Alibaba Cloud DashScope Console](https://dashscope.console.aliyun.com/)
2. Create or obtain your API key (format: `sk-...`)
3. Save the key securely

Alternatively, set the key as an environment variable:
```bash
export DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"
```

### ⚠️ ВАЖНО: Требование способа оплаты / IMPORTANT: Payment Method Requirement

**Для работы с DashScope ключом (sk-...) и любыми моделями WAN 2.5 необходимо добавить способ оплаты (банковская карта, AliPay и др.) в панели Alibaba Cloud. Без этого API-ключ не работает, и все обращения заблокированы: будет ошибка `NO_AVAILABLE_PAYMENT_METHOD`. Даже бесплатные квоты требуют актуальной оплаты! Проверьте ваш аккаунт перед запуском.**

**To use DashScope API keys (sk-...) with any WAN 2.5 models, you MUST add a payment method (bank card, AliPay, etc.) to your Alibaba Cloud account. Without this, the API key will not work and all requests will be blocked with error `NO_AVAILABLE_PAYMENT_METHOD`. Even free quotas require an active payment method! Please verify your account before running.**

## ▶️ Running
```bash
python main.py
```

After starting, open your browser at:
- Locally: http://localhost:7860
- From network: http://YOUR_IP:7860

## 📖 Usage

### Text2Video Mode
1. Enter your DashScope API key in the "DashScope API Key" field
2. Select "text2video" mode
3. Enter a text description (prompt)
4. Configure generation parameters (duration, resolution, FPS, seed)
5. Click "Generate Video" and wait for completion
6. The video URL will be provided upon completion

### Img2Video Mode
1. Enter your DashScope API key in the "DashScope API Key" field
2. Select "img2video" mode
3. Upload an image from your PC (preferred) OR provide an image URL
   - Image upload field will appear when img2video is selected
   - Local file upload has priority over URL if both are provided
4. Enter a text description (prompt) describing what should happen in the video
5. Configure generation parameters (duration, resolution, FPS, seed)
6. Click "Generate Video" and wait for completion
7. The video URL will be provided upon completion

Note: In img2video mode, the uploaded/provided image will be used as the first frame of the generated video.

## 🗂️ Project Structure
```
WanSuper/
├── main.py              # Gradio UI with mode selection and image upload
├── wan_api.py           # DashScopeClient class (async_call/fetch/wait)
├── requirements.txt     # Project dependencies (includes dashscope)
└── README.md            # Documentation
```

## 🔧 Components

### main.py
Gradio application with:
- DashScope API key input
- Mode selection (text2video / img2video)
- Local image upload support
- Generation parameters
- Integrates DashScopeClient for video generation

### wan_api.py
DashScope WAN 2.5 API client using official SDK:

```python
from wan_api import DashScopeClient

# Initialize client
client = DashScopeClient(api_key="sk-...")

# Submit generation task (async_call)
task_id = client.submit_generation(
    prompt="A beautiful sunset over the ocean",
    image_url="https://example.com/image.jpg",  # Optional (URL or file path)
    duration=5,
    width=1280,
    height=720,
    fps=24,
    seed=42  # Optional
)

# Check status (fetch)
status_info = client.check_status(task_id)

# Wait for completion (wait)
video_url = client.wait_for_completion(
    task_id,
    progress_callback=lambda p: print(f"Progress: {p*100}%")
)

print(f"Video URL: {video_url}")
```

Key methods:
- **submit_generation()** - Submit task using dashscope.VideoSynthesis.async_call()
  - Handles both URL and local file paths for images
- **check_status()** - Check task status using dashscope.VideoSynthesis.fetch()
- **wait_for_completion()** - Wait for task and return video URL

Status mapping:
- DashScope PENDING → pending
- DashScope RUNNING → processing
- DashScope SUCCEEDED → completed
- DashScope FAILED → failed

## 🌐 DashScope API Pattern

The official DashScope SDK follows this pattern:

1. **async_call** - Submit generation task
   ```python
   response = dashscope.VideoSynthesis.async_call(
       model="wan25-turbo",
       input={"prompt": "..."},
       parameters={"duration": 5, "size": "1280x720"}
   )
   task_id = response.output.get("task_id")
   ```

2. **fetch** - Check task status
   ```python
   response = dashscope.VideoSynthesis.fetch(task_id=task_id)
   status = response.output.get("task_status")  # PENDING, RUNNING, SUCCEEDED, FAILED
   ```

3. **wait** - Poll until completion
   ```python
   while status != "SUCCEEDED":
       response = dashscope.VideoSynthesis.fetch(task_id=task_id)
       status = response.output.get("task_status")
       time.sleep(5)
   video_url = response.output.get("video_url")
   ```

## 💡 Tips
- Store API key in environment variables (`DASHSCOPE_API_KEY`), don't commit it
- **Mode Selection**: Choose the appropriate mode for your use case
  - `text2video`: Generate video purely from text description
  - `img2video`: Animate from a starting image
- **Image Upload**: Local file upload has priority over URL
  - Supports common image formats (JPEG, PNG, etc.)
  - Image will be used as the first frame
- Detailed prompts improve generation quality
- Higher resolutions require more time and API quota
- Use seed for reproducible results
- API key format: `sk-` followed by alphanumeric characters

## 🐛 Troubleshooting
- **"API key required"** - Provide valid DashScope API key
- **"Failed to submit"** - Check API key validity and WAN 2.5 access
- **"Generation failed"** - Check API quota/limits and parameter formats
- **"img2video mode requires an image"** - Upload a file or provide URL when using img2video mode
- **Slow generation** - Reduce duration/resolution/FPS
- **Import errors** - Run `pip install dashscope>=1.14.0`
- **Image not uploading** - Check file format and size, try using a URL instead

## 📚 Documentation
- [DashScope Official Documentation](https://help.aliyun.com/zh/dashscope/)
- [WAN 2.5 Video Synthesis API](https://help.aliyun.com/zh/dashscope/developer-reference/api-details-9)
- [Gradio Documentation](https://gradio.app/docs/)

## 📝 License
Open source project for use and modification.

## 📧 Contact
For issues or questions, create an Issue in the repository.
