import requests
import time
import os
from typing import Optional, Dict, Any, Callable


class WANClient:
    """
    Клиент для взаимодействия с Alibaba WAN 2.5 Cloud API
    """

    def __init__(self, api_key: str, api_url: str = "https://api.aliyun.com/wan/v2.5"):
        """
        Инициализация клиента

        Args:
            api_key: API-ключ Alibaba WAN 2.5
            api_url: URL облачного сервера WAN 2.5
        """
        self.api_url = api_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {api_key}'
        })

    def check_health(self) -> bool:
        """
        Проверка доступности сервера

        Returns:
            True если сервер доступен, False в противном случае
        """
        try:
            response = self.session.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Ошибка проверки доступности: {e}")
            return False

    def submit_generation(
        self,
        prompt: str,
        duration: int = 5,
        width: int = 1280,
        height: int = 720,
        fps: int = 24,
        seed: Optional[int] = None,
    ) -> Optional[str]:
        """
        Отправка запроса на генерацию видео

        Args:
            prompt: Текстовое описание для генерации видео
            duration: Продолжительность видео в секундах
            width: Ширина видео в пикселях
            height: Высота видео в пикселях
            fps: Количество кадров в секунду
            seed: Seed для воспроизводимости результатов

        Returns:
            ID задачи или None в случае ошибки
        """
        try:
            payload = {
                "prompt": prompt,
                "duration": duration,
                "width": width,
                "height": height,
                "fps": fps,
            }

            if seed is not None and seed >= 0:
                payload["seed"] = seed

            response = self.session.post(
                f"{self.api_url}/api/generate",
                json=payload,
                timeout=30,
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("task_id")
            else:
                print(
                    f"Ошибка при отправке запроса: {response.status_code} - {response.text}"
                )
                return None

        except Exception as e:
            print(f"Ошибка при отправке запроса на генерацию: {e}")
            return None

    def check_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Проверка статуса задачи генерации

        Args:
            task_id: ID задачи

        Returns:
            Информация о статусе задачи или None в случае ошибки
        """
        try:
            response = self.session.get(
                f"{self.api_url}/api/status/{task_id}",
                timeout=10,
            )

            if response.status_code == 200:
                return response.json()
            else:
                print(f"Ошибка проверки статуса: {response.status_code}")
                return None

        except Exception as e:
            print(f"Ошибка при проверке статуса: {e}")
            return None

    def download_video(self, task_id: str, output_path: Optional[str] = None) -> Optional[str]:
        """
        Загрузка сгенерированного видео

        Args:
            task_id: ID задачи
            output_path: Путь для сохранения видео (опционально)

        Returns:
            Путь к загруженному файлу или None в случае ошибки
        """
        try:
            response = self.session.get(
                f"{self.api_url}/api/download/{task_id}",
                timeout=120,
                stream=True,
            )

            if response.status_code == 200:
                # Определение пути для сохранения
                if output_path is None:
                    os.makedirs("output", exist_ok=True)
                    output_path = f"output/video_{task_id}.mp4"

                # Сохранение файла
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                return output_path
            else:
                print(f"Ошибка загрузки видео: {response.status_code}")
                return None

        except Exception as e:
            print(f"Ошибка при загрузке видео: {e}")
            return None

    def wait_for_completion(
        self,
        task_id: str,
        max_wait_time: int = 1200,
        poll_interval: int = 3,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Optional[str]:
        """
        Ожидание завершения генерации и загрузка видео

        Args:
            task_id: ID задачи
            max_wait_time: Максимальное время ожидания в секундах
            poll_interval: Интервал проверки статуса в секундах
            progress_callback: Функция обратного вызова для отчета о прогрессе

        Returns:
            Путь к загруженному видео или None в случае ошибки
        """
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            status_info = self.check_status(task_id)

            if status_info is None:
                time.sleep(poll_interval)
                continue

            status = status_info.get("status")
            progress = status_info.get("progress", 0)

            # Обновление прогресса
            if progress_callback:
                try:
                    progress_callback(float(progress) / 100.0)
                except Exception:
                    pass

            if status == "completed":
                # Загрузка готового видео
                return self.download_video(task_id)
            elif status == "failed":
                error = status_info.get("error", "Неизвестная ошибка")
                print(f"Генерация не удалась: {error}")
                return None
            elif status in ["pending", "processing"]:
                # Продолжаем ждать
                time.sleep(poll_interval)
            else:
                print(f"Неизвестный статус: {status}")
                time.sleep(poll_interval)

        print(f"Превышено время ожидания ({max_wait_time} сек)")
        return None


def test_connection(api_key: str, api_url: str = "https://api.aliyun.com/wan/v2.5") -> bool:
    """
    Тестовая функция для проверки подключения к серверу

    Args:
        api_key: API-ключ Alibaba WAN 2.5
        api_url: URL сервера WAN 2.5

    Returns:
        True если подключение успешно, False в противном случае
    """
    client = WANClient(api_key=api_key, api_url=api_url)
    return client.check_health()


if __name__ == "__main__":
    # Пример использования
    API_KEY = os.environ.get("ALIWAN_API_KEY", "")
    API_URL = "https://api.aliyun.com/wan/v2.5"

    if not API_KEY:
        print("❌ Установите переменную окружения ALIWAN_API_KEY с вашим API-ключом")
        raise SystemExit(1)

    print("Тестирование подключения к WAN 2.5 (Cloud)...")
    if test_connection(API_KEY, API_URL):
        print("✅ Подключение успешно!")

        # Пример генерации видео
        client = WANClient(API_KEY, API_URL)
        print("\nЗапуск генерации тестового видео...")

        task_id = client.submit_generation(
            prompt="A beautiful sunset over the ocean with birds flying in the sky",
            duration=5,
            width=1280,
            height=720,
            fps=24,
        )

        if task_id:
            print(f"Задача создана: {task_id}")
            print("Ожидание завершения генерации...")

            video_path = client.wait_for_completion(task_id)

            if video_path:
                print(f"✅ Видео успешно сгенерировано: {video_path}")
            else:
                print("❌ Не удалось сгенерировать видео")
        else:
            print("❌ Не удалось создать задачу")
    else:
        print(f"❌ Не удалось подключиться к серверу: {API_URL}")
        print("Убедитесь, что облачный WAN 2.5 доступен и API-ключ корректен.")
