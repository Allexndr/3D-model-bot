import asyncio
import os
import re
import requests
from typing import List, Dict, Optional
from telegram import Bot
from telegram.error import TelegramError
import time

from config import BOT_TOKEN, TARGET_GROUP_USERNAME
from database import ModelsDatabase
from image_analyzer import ImageAnalyzer

class GroupParser:
    def __init__(self):
        self.bot = Bot(token=BOT_TOKEN)
        self.database = ModelsDatabase()
        self.image_analyzer = ImageAnalyzer()
        self.images_dir = "downloaded_images"
        
        # Создание папки для изображений
        os.makedirs(self.images_dir, exist_ok=True)
    
    async def parse_group_messages(self, group_username: str, limit: int = 100):
        """Парсинг сообщений из группы"""
        try:
            print(f"Начинаю парсинг группы {group_username}...")
            
            # Получение информации о группе
            chat = await self.bot.get_chat(group_username)
            group_id = str(chat.id)
            
            print(f"ID группы: {group_id}")
            
            # Получение метаданных о последнем парсинге
            metadata = self.database.get_group_metadata(group_id)
            last_message_id = metadata['last_parsed_message_id'] if metadata else 0
            
            print(f"Последний обработанный message_id: {last_message_id}")
            
            processed_count = 0
            current_message_id = last_message_id + 1 if last_message_id > 0 else 1
            
            while processed_count < limit:
                try:
                    # Получение сообщения по ID
                    message = await self.bot.forward_message(
                        chat_id=chat.id,
                        from_chat_id=chat.id,
                        message_id=current_message_id
                    )
                    
                    # Проверка, есть ли в сообщении изображения
                    if message.photo:
                        await self.process_message_with_image(message, group_id)
                        processed_count += 1
                        print(f"Обработано {processed_count} сообщений")
                    
                    # Обновление метаданных
                    self.database.update_group_metadata(group_id, current_message_id)
                    current_message_id += 1
                    
                    # Пауза между запросами для избежания лимитов
                    await asyncio.sleep(0.1)
                    
                except TelegramError as e:
                    if "Message not found" in str(e):
                        current_message_id += 1
                        continue
                    elif "Too Many Requests" in str(e):
                        print("Превышен лимит запросов, ждем...")
                        await asyncio.sleep(30)
                        continue
                    else:
                        print(f"Ошибка Telegram API: {e}")
                        break
                
                except Exception as e:
                    print(f"Неожиданная ошибка: {e}")
                    current_message_id += 1
                    continue
            
            print(f"Парсинг завершен. Обработано {processed_count} сообщений с изображениями.")
            
        except Exception as e:
            print(f"Ошибка при парсинге группы: {e}")
    
    async def process_message_with_image(self, message, group_id: str):
        """Обработка сообщения с изображением"""
        try:
            # Получение самого большого размера фотографии
            photo = message.photo[-1]
            
            # Создание ссылки на пост
            post_link = f"https://t.me/blocks_01/{message.message_id}"
            
            # Извлечение заголовка и описания из текста сообщения
            title, description = self.extract_title_and_description(message.caption or "")
            
            # Скачивание изображения
            image_path = await self.download_image(photo, message.message_id)
            
            if image_path:
                # Получение эмбеддинга изображения
                with open(image_path, 'rb') as f:
                    image_bytes = f.read()
                
                embedding = self.image_analyzer.process_image_from_bytes(image_bytes)
                
                if embedding is not None:
                    # Сохранение в базу данных
                    self.database.save_model(
                        message_id=message.message_id,
                        group_id=group_id,
                        post_link=post_link,
                        title=title,
                        description=description,
                        image_path=image_path,
                        embedding=embedding
                    )
                    
                    print(f"Модель сохранена: {title[:30]}...")
                
        except Exception as e:
            print(f"Ошибка при обработке сообщения {message.message_id}: {e}")
    
    async def download_image(self, photo, message_id: int) -> Optional[str]:
        """Скачивание изображения"""
        try:
            # Получение файла
            file = await self.bot.get_file(photo.file_id)
            
            # Создание пути для сохранения
            file_extension = file.file_path.split('.')[-1] if '.' in file.file_path else 'jpg'
            image_path = os.path.join(self.images_dir, f"model_{message_id}.{file_extension}")
            
            # Скачивание файла
            response = requests.get(file.file_url, timeout=30)
            response.raise_for_status()
            
            with open(image_path, 'wb') as f:
                f.write(response.content)
            
            return image_path
            
        except Exception as e:
            print(f"Ошибка при скачивании изображения: {e}")
            return None
    
    def extract_title_and_description(self, text: str) -> tuple[str, str]:
        """Извлечение заголовка и описания из текста сообщения"""
        if not text:
            return "3D Модель", ""
        
        # Очистка текста
        text = text.strip()
        
        # Попытка найти заголовок (первая строка или до первого переноса)
        lines = text.split('\n')
        title = lines[0] if lines else "3D Модель"
        
        # Ограничение длины заголовка
        if len(title) > 50:
            title = title[:47] + "..."
        
        # Описание - оставшийся текст
        description = '\n'.join(lines[1:]) if len(lines) > 1 else ""
        
        # Удаление лишних символов и ссылок
        title = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', title)
        title = re.sub(r'[#@]', '', title).strip()
        
        if not title:
            title = "3D Модель"
        
        return title, description
    
    async def update_database(self, limit: int = 500):
        """Обновление базы данных новыми моделями"""
        print("Начинаю обновление базы данных...")
        await self.parse_group_messages(TARGET_GROUP_USERNAME, limit)
        
        models_count = self.database.get_models_count()
        print(f"Общее количество моделей в базе: {models_count}")

# Функция для запуска парсера
async def run_parser():
    parser = GroupParser()
    await parser.update_database(limit=100)

if __name__ == "__main__":
    asyncio.run(run_parser()) 