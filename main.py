import gradio as gr
import os
from wan_api import WANClient

def generate_video(api_url, prompt, duration, resolution, fps, seed, progress=gr.Progress()):
    """
    Генерация видео через WAN 2.5 API
    
    Args:
        api_url: URL сервера WAN 2.5 (например, http://192.168.1.100:8000)
        prompt: Текстовое описание для генерации видео
        duration: Продолжительность видео в секундах
        resolution: Разрешение видео (формат: "1920x1080")
        fps: Количество кадров в секунду
        seed: Seed для воспроизводимости результатов
    
    Returns:
        Путь к сгенерированному видео или сообщение об ошибке
    """
    try:
        progress(0, desc="Инициализация...")
        
        # Создание клиента
        client = WANClient(api_url)
        
        progress(0.1, desc="Подключение к серверу...")
        
        # Проверка доступности сервера
        if not client.check_health():
            return None, "❌ Ошибка: Сервер недоступен. Проверьте URL и доступность сервиса."
        
        progress(0.2, desc="Отправка запроса на генерацию...")
        
        # Разбор разрешения
        width, height = map(int, resolution.split('x'))
        
        # Отправка запроса на генерацию
        task_id = client.submit_generation(
            prompt=prompt,
            duration=duration,
            width=width,
            height=height,
            fps=fps,
            seed=seed if seed > 0 else None
        )
        
        if not task_id:
            return None, "❌ Ошибка: Не удалось отправить запрос на генерацию."
        
        progress(0.3, desc=f"Генерация видео (ID: {task_id})...")
        
        # Ожидание завершения генерации
        video_path = client.wait_for_completion(task_id, progress_callback=lambda p: progress(0.3 + p * 0.6, desc="Генерация видео..."))
        
        if video_path:
            progress(1.0, desc="Готово!")
            return video_path, f"✅ Видео успешно сгенерировано! (ID: {task_id})"
        else:
            return None, "❌ Ошибка: Не удалось сгенерировать видео."
            
    except Exception as e:
        return None, f"❌ Ошибка: {str(e)}"

def create_interface():
    """
    Создание Gradio интерфейса
    """
    with gr.Blocks(title="WAN 2.5 Video Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎬 WAN 2.5 Video Generator
            
            Генерация видео с помощью WAN 2.5 по локальной или удаленной сети.
            
            ## Как использовать:
            1. Укажите URL сервера WAN 2.5 (например: `http://192.168.1.100:8000`)
            2. Введите текстовое описание желаемого видео
            3. Настройте параметры генерации
            4. Нажмите "Сгенерировать видео"
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Настройки подключения")
                api_url = gr.Textbox(
                    label="URL сервера WAN 2.5",
                    placeholder="http://192.168.1.100:8000",
                    value="http://localhost:8000",
                    info="Адрес сервера с установленным WAN 2.5"
                )
                
                gr.Markdown("### 📝 Параметры генерации")
                prompt = gr.Textbox(
                    label="Описание видео (Prompt)",
                    placeholder="A serene sunset over the ocean with birds flying...",
                    lines=3,
                    info="Опишите, что вы хотите увидеть в видео"
                )
                
                with gr.Row():
                    duration = gr.Slider(
                        label="Длительность (сек)",
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
                    label="Разрешение",
                    choices=["512x512", "768x768", "1024x576", "1280x720", "1920x1080"],
                    value="1280x720"
                )
                
                seed = gr.Number(
                    label="Seed (опционально)",
                    value=-1,
                    precision=0,
                    info="Используйте -1 для случайного seed"
                )
                
                generate_btn = gr.Button("🎬 Сгенерировать видео", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                gr.Markdown("### 🎥 Результат")
                output_video = gr.Video(label="Сгенерированное видео")
                output_status = gr.Textbox(label="Статус", lines=2)
                
                gr.Markdown(
                    """
                    ### 💡 Советы:
                    - Используйте детальные описания для лучших результатов
                    - Большие разрешения требуют больше времени и ресурсов
                    - Seed позволяет воспроизводить одинаковые результаты
                    """
                )
        
        # Привязка функции генерации к кнопке
        generate_btn.click(
            fn=generate_video,
            inputs=[api_url, prompt, duration, resolution, fps, seed],
            outputs=[output_video, output_status]
        )
        
        gr.Markdown(
            """
            ---
            
            ### 📚 Документация
            - Убедитесь, что сервер WAN 2.5 запущен и доступен
            - Проверьте, что указан правильный IP-адрес и порт
            - При проблемах проверьте логи сервера
            """
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",  # Доступ из локальной сети
        server_port=7860,
        share=False,  # Установите True для публичного доступа через Gradio
        show_error=True
    )
