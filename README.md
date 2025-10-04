# 🎬 WAN 2.5 Video Generator

> **Создано [The Angel Studio](https://boosty.to/the_angel)**  
> 🎨 Professional AI Video Generation Tools

[![Boosty](https://img.shields.io/badge/Support-Boosty-orange)](https://boosty.to/the_angel/donate)

---

## 📢 About The Angel Studio

**The Angel Studio** — студия разработки AI-инструментов для креативных профессионалов.

- 🌟 **Поддержать проект**: [https://boosty.to/the_angel](https://boosty.to/the_angel)
- 💬 **Обратная связь**: Создавайте Issues в этом репозитории
- 🚀 **Больше проектов**: Следите за обновлениями на Boosty

---

## 🎯 О проекте

**WAN Super** — профессиональное приложение для генерации видео с использованием:
- **Облачного API** (Alibaba DashScope WAN 2.5)
- **Локального сервера** (WAN 2.5 self-hosted)

### ✨ Новая архитектура проекта

- 🏠 **Главная страница с меню** — выбор между API и локальным режимом
- 📁 **Структурированный код** — API-логика в папке `api/`
- 🎨 **Брендирование The Angel Studio**
- 🔧 **Модульная архитектура** для легкого расширения

---

## 🚀 Быстрый старт

### Установка

```bash
git clone https://github.com/sanek1989/WanSuper.git
cd WanSuper
```

#### 🔥 Установка PyTorch (обязательно для локального режима)

**PyTorch требуется для работы с GPU и локальной генерации видео.**

Выберите версию на официальном сайте: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

**Пример для CUDA 11.8 (Windows/Linux):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Для CPU (если нет GPU):**
```bash
pip install torch torchvision torchaudio
```

#### Установка остальных зависимостей

```bash
pip install -r requirements.txt
```

### Запуск

```bash
python main.py
```

Откройте браузер:
- Локально: [http://localhost:7860](http://localhost:7860)
- Из сети: `http://YOUR_IP:7860`

---

##  🎮 Главное меню

При запуске вы увидите **главную страницу The Angel Studio** с выбором режима:

### 1️⃣ **API Mode** (Облачная генерация)

- Использует Alibaba Cloud DashScope WAN 2.5 API
- Требует API ключ (формат: `sk-...`)
- Поддержка text2video и img2video
- Быстрая генерация без локального GPU

### 2️⃣ **Local Mode** (Локальный сервер)

- Использует локально развернутый WAN 2.5
- Не требует облачных ключей
- Полный контроль и приватность
- Требует мощное GPU оборудование

---

## 📁 Структура проекта

```
WanSuper/
├── main.py                 # Главная страница с меню выбора режима
├── api/                    # API-режим (облачная генерация)
│   ├── wan_api.py          # DashScope API клиент
│   └── api_interface.py    # Gradio интерфейс для API
├── local/                  # Локальный режим (self-hosted)
│   └── local_interface.py  # Gradio интерфейс для локального сервера
├── requirements.txt        # Зависимости
└── README.md               # Документация
```

---

## 🔐 API Mode — Облачная генерация

### Получение DashScope API ключа

1. Посетите [Alibaba Cloud DashScope Console](https://dashscope.console.aliyun.com/)
2. Создайте или получите API ключ (формат: `sk-...`)
3. Сохраните ключ безопасно

### ⚠️ ВАЖНО: Требование способа оплаты

**Для работы с DashScope ключом необходимо добавить способ оплаты** (банковская карта, AliPay и др.) в Alibaba Cloud. Без этого API-ключ не работает, будет ошибка `NO_AVAILABLE_PAYMENT_METHOD`. Даже бесплатные квоты требуют активной оплаты!

**To use DashScope API keys, you MUST add a payment method** (bank card, AliPay, etc.) to your Alibaba Cloud account. Without this, the API key will not work and requests will be blocked with error `NO_AVAILABLE_PAYMENT_METHOD`. Even free quotas require an active payment method!

### Возможности API Mode

- ✅ **text2video** — генерация видео из текстового описания
- ✅ **img2video** — анимация изображения (первый кадр)
- ✅ Загрузка локальных изображений или URL
- ✅ Настройка параметров (длительность, разрешение, FPS, seed)
- ✅ Отслеживание прогресса
- ✅ Прямая ссылка на видео

---

## 🏠 Local Mode — Локальный сервер

### Требования

- Локально развернутый WAN 2.5 сервер
- Мощное GPU (рекомендуется 24GB+ VRAM)
- Python 3.8+

### Возможности Local Mode

- ✅ Полная приватность данных
- ✅ Не требует облачных API ключей
- ✅ Локальная генерация без лимитов
- ✅ Полный контроль над процессом

> **Примечание:** Режим Local Mode находится в разработке. Следите за обновлениями!

---

## 🎨 The Angel Studio Features

### В главном интерфейсе

- 🏢 Брендирование The Angel Studio
- 🎨 Стильный дизайн с Gradio
- 🔄 Легкое переключение между режимами
- 📊 Информация о проекте и поддержке

### Ссылки

- **Поддержать студию:** [https://boosty.to/the_angel](https://boosty.to/the_angel)
- **GitHub:** [https://github.com/sanek1989/WanSuper](https://github.com/sanek1989/WanSuper)

---

## 💡 Tips

- 🔒 Храните API ключи в переменных окружения (`DASHSCOPE_API_KEY`)
- 📝 Детализированные промпты улучшают качество
- ⚙️ Высокое разрешение требует больше времени и ресурсов
- 🎲 Используйте seed для воспроизводимых результатов
- 📁 Локальные файлы изображений имеют приоритет над URL

---

## 🐛 Troubleshooting

### API Mode

- "API key required" — Укажите валидный DashScope API ключ
- "Failed to submit" — Проверьте валидность ключа и доступ к WAN 2.5
- "NO_AVAILABLE_PAYMENT_METHOD" — Добавьте способ оплаты в Alibaba Cloud

### Local Mode

- Проверьте, что локальный WAN 2.5 сервер запущен
- Убедитесь в наличии достаточных GPU ресурсов

### Общие проблемы

- Медленная генерация — Снизьте длительность/разрешение/FPS
- Ошибки импорта — Выполните `pip install -r requirements.txt`
- Изображение не загружается — Проверьте формат и размер файла

---

## 📚 Documentation

- [DashScope Official Documentation](https://help.aliyun.com/zh/dashscope/)
- [WAN 2.5 Video Synthesis API](https://help.aliyun.com/zh/dashscope/developer-reference/api-details-9)
- [Gradio Documentation](https://gradio.app/docs/)

---

## 📝 License

Open source project created by **The Angel Studio**.  
Free for use and modification.

---

## 📧 Contact

- 💬 **Issues:** Create an Issue in this repository
- 🎨 **Support:** [https://boosty.to/the_angel](https://boosty.to/the_angel)

---

**Made with ❤️ by [The Angel Studio](https://boosty.to/the_angel)**
