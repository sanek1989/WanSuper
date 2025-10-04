"""WAN Super Video/image Generator - Main Landing Page
The Angel Studio - Professional AI Video Generation Tools
"""
import gradio as gr
from api.api_interface import create_api_interface
from local.local_interface import create_local_interface

def create_greeting_content():
    """Create greeting/welcome content"""
    return gr.Markdown(
        """
        <div style="text-align: center; padding: 60px 40px;">
            <h1 style="font-size: 4em; margin-bottom: 20px; background: linear-gradient(45deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
                🎬 THE ANGEL STUDIO
            </h1>
            <h2 style="font-size: 2.5em; margin-bottom: 30px; color: #34495e;">
                WAN 2.5 Video Generator
            </h2>
            <p style="font-size: 1.4em; color: #7f8c8d; margin-bottom: 40px;">
                Professional AI Video Generation Tools
            </p>
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 15px; color: white; margin: 40px 0;">
                <h3 style="margin-bottom: 20px; font-size: 1.5em;">🌟 Добро пожаловать!</h3>
                <p style="font-size: 1.1em; line-height: 1.6;">Выберите режим генерации видео из левого меню. Мы предлагаем облачную генерацию через API или локальное развертывание для максимальной приватности.</p>
            </div>
        </div>
        """
    )

def create_about_content():
    """Create about project content"""
    return gr.Markdown(
        """
        # 📖 О проекте WAN Super
        
        **WAN Super** — это профессиональное приложение для генерации видео с использованием современных AI-технологий.
        
        ## 🎯 Основные возможности
        
        ### 🔑 API Mode (Облачная генерация)
        - ✅ Использует Alibaba Cloud WAN 2.5 API
        - ✅ Не требует локального GPU
        - ✅ Быстрая генерация
        - ✅ Поддержка text2video и img2video
        - ⚠️ Требует API ключ (формат: sk-...)
        
        ### 🏠 Local Mode (Локальный сервер)
        - ✅ Полная приватность данных
        - ✅ Не требует облачных API ключей
        - ✅ Локальная генерация без лимитов
        - ✅ Полный контроль над процессом
        - ⚠️ Требует мощное GPU (4GB+ VRAM)
        
        ## 🛠️ Технические особенности
        
        - **text2video**: Генерация видео из текстовых описаний
        - **img2video**: Анимация изображений в видео
        - **Высокое качество**: Поддержка различных разрешений и частот кадров
        - **Гибкость**: Настраиваемая длительность, разрешение, FPS и seed
        
        ## 📁 Архитектура проекта
        
        ```
        WanSuper/
        ├── main.py                 # Главная страница с меню
        ├── api/                    # API-режим (облачная генерация)
        │   ├── wan_api.py          # DashScope API клиент
        │   └── api_interface.py    # Gradio интерфейс для API
        ├── local/                  # Локальный режим
        │   └── local_interface.py  # Градио интерфейс для локального сервера
        ├── requirements.txt        # Зависимости
        └── README.md               # Документация
        ```
        
        ---
        
        <div style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 10px; margin: 20px 0;">
            <p style="color: #666; font-style: italic;">Made with ❤️ by The Angel Studio</p>
        </div>
        """
    )

def create_support_content():
    """Create support content"""
    return gr.Markdown(
        """
        # 🌟 Поддержать The Angel Studio
        
        Если вам нравится наш проект, вы можете поддержать его развитие!
        
        ## 💝 Способы поддержки
        
        ### 🎨 Boosty
        Основная платформа для поддержки нашей студии:
        
        **[👉 Поддержать на Boosty](https://boosty.to/the_angel)**
        
        ### 🐛 GitHub
        Помогите улучшить проект:
        - Создавайте Issues для отчетов об ошибках
        - Предлагайте новые функции
        - Делитесь отзывами
        
        **[👉 GitHub Repository](https://github.com/sanek1989/WanSuper)**
        
        ## 🚀 Планы развития
        
        - 🔄 Расширение поддержки локальных моделей
        - 🎨 Улучшение пользовательского интерфейса
        - 📱 Мобильная версия
        - 🌐 Поддержка дополнительных языков
        - 🔧 Новые настройки и параметры генерации
        
        ## 📞 Контакты
        
        - **💬 Обратная связь**: Создавайте Issues в GitHub репозитории
        - **🎨 Поддержка**: [Boosty](https://boosty.to/the_angel)
        - **📚 Документация**: README.md в репозитории
        
        ---
        
        <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; margin: 30px 0;">
            <h3 style="margin-bottom: 15px;">🙏 Спасибо за поддержку!</h3>
            <p style="font-size: 1.1em; margin: 0;">Ваша поддержка помогает нам создавать лучшие AI-инструменты для креативных профессионалов</p>
        </div>
        """
    )

def create_main_interface():
    """Create main interface with sidebar navigation"""
    
    # Define the navigation structure
    nav_items = [
        ("🏠 Главная", "home"),
        ("🔑 Облачный режим (API)", "api"),
        ("🏠 Локальный режим", "local"),
        ("📖 О проекте", "about"),
        ("🌟 Поддержать", "support")
    ]
    
    with gr.Blocks(
        title="The Angel Studio - WAN Super",
        theme=gr.themes.Soft()
    ) as demo:
        
        with gr.Row():
            # Left sidebar with navigation
            with gr.Column(scale=1, min_width=250):
                gr.Markdown(
                    """
                    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white; margin-bottom: 20px;">
                        <h3 style="margin: 0; font-size: 1.2em;">🎬 THE ANGEL STUDIO</h3>
                        <p style="margin: 5px 0 0 0; font-size: 0.9em; opacity: 0.9;">WAN Super Generator</p>
                    </div>
                    """
                )
                
                # Navigation buttons
                nav_buttons = {}
                for label, key in nav_items:
                    nav_buttons[key] = gr.Button(
                        label,
                        variant="primary" if key == "home" else "secondary",
                        size="lg",
                        elem_classes=["nav-button"]
                    )
            
            # Main content area
            with gr.Column(scale=3):
                # Content containers (all hidden initially except home)
                content_components = {}
                
                with gr.Column(visible=True) as home_content:
                    content_components["home"] = create_greeting_content()
                
                with gr.Column(visible=False) as api_content:
                    content_components["api"] = create_api_interface()
                
                with gr.Column(visible=False) as local_content:
                    content_components["local"] = create_local_interface()
                
                with gr.Column(visible=False) as about_content:
                    content_components["about"] = create_about_content()
                
                with gr.Column(visible=False) as support_content:
                    content_components["support"] = create_support_content()
                
                # Map content containers
                content_containers = {
                    "home": home_content,
                    "api": api_content,
                    "local": local_content,
                    "about": about_content,
                    "support": support_content
                }
        
        def switch_section(section_key):
            """Switch to the selected section"""
            updates = []
            
            # Hide all content containers and show only the selected one
            for key in ["home", "api", "local", "about", "support"]:
                updates.append(gr.update(visible=(key == section_key)))
            
            # Update button variants
            for key in ["home", "api", "local", "about", "support"]:
                updates.append(gr.update(
                    variant="primary" if key == section_key else "secondary"
                ))
            
            return updates
        
        # Set up navigation click handlers
        for key, button in nav_buttons.items():
            button.click(
                fn=lambda k=key: switch_section(k),
                inputs=[],
                outputs=[
                    home_content,
                    api_content,
                    local_content,
                    about_content,
                    support_content,
                    nav_buttons["home"],
                    nav_buttons["api"],
                    nav_buttons["local"],
                    nav_buttons["about"],
                    nav_buttons["support"]
                ]
            )
    
    return demo

if __name__ == "__main__":
    print("✨ Starting The Angel Studio - WAN Super Video/image Generator...")
    print("🌐 Server will be available at: http://localhost:7860")
    print("🌟 Support us: https://boosty.to/the_angel")
    
    demo = create_main_interface()
    
    demo.launch(
        server_name="localhost",
        server_port=7860,
        share=False,
        show_error=True
    )
