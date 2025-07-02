#!/usr/bin/env python3
"""
Автообновляющая система для мониторинга новых 3D моделей
Постоянно отслеживает канал @blocks_01 и добавляет новые модели
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List
from channel_parser import ChannelParser
from database_factory import get_database
from config import TARGET_GROUP_USERNAME

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('auto_updater.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutoUpdater:
    def __init__(self):
        self.parser = ChannelParser()
        self.db = get_database()
        self.check_interval = 1800  # 30 минут в секундах
        self.last_message_id = None
        
    async def get_last_processed_message_id(self) -> int:
        """Получить ID последнего обработанного сообщения из базы"""
        try:
            # Получаем максимальный message_id из базы данных
            # Это будет отправная точка для поиска новых сообщений
            max_id = self.db.get_max_message_id()
            if max_id:
                logger.info(f"📊 Последнее обработанное сообщение: {max_id}")
                return max_id
            else:
                logger.info("📊 Первый запуск - начинаем с самых новых сообщений")
                return None
        except Exception as e:
            logger.error(f"❌ Ошибка получения последнего ID: {e}")
            return None

    async def check_for_new_models(self) -> Dict[str, int]:
        """Проверка новых моделей в канале"""
        try:
            logger.info(f"🔍 Проверяю новые модели в {TARGET_GROUP_USERNAME}...")
            
            # Аутентификация
            if not await self.parser.authenticate():
                return {'error': 'authentication_failed', 'new_models': 0}

            # Получаем ID последнего обработанного сообщения
            last_id = await self.get_last_processed_message_id()
            
            new_models = []
            new_count = 0
            
            # Получаем только новые сообщения (после last_id)
            async for message in self.parser.client.iter_messages(
                TARGET_GROUP_USERNAME, 
                min_id=last_id if last_id else None,
                limit=1000  # Максимум 1000 новых сообщений за раз
            ):
                try:
                    # Проверка наличия медиа
                    if not message.media:
                        continue
                    
                    # Проверка типа медиа (фото или документ-изображение)
                    is_image = False
                    if hasattr(message.media, 'photo'):
                        is_image = True
                    elif hasattr(message.media, 'document'):
                        if message.media.document.mime_type and message.media.document.mime_type.startswith('image/'):
                            is_image = True
                    
                    if not is_image:
                        continue
                    
                    # Проверка на дублирование
                    existing = self.db.get_model_by_message_id(message.id)
                    if existing:
                        continue
                    
                    # Извлечение текста
                    text = message.message or ""
                    
                    # Проверка на 3D модель
                    text_lower = text.lower()
                    model_keywords = ['3d', 'модель', 'model', 'max', 'obj', 'fbx', 'архитектура', 'мебель', 'decor']
                    
                    if not any(keyword in text_lower for keyword in model_keywords) and len(text) < 10:
                        continue
                    
                    # Анализ изображения
                    logger.info(f"🖼️ Анализирую новое изображение из сообщения {message.id}...")
                    embedding = await self.parser.download_and_analyze_image(message)
                    
                    if embedding is None:
                        continue
                    
                    # Извлечение информации о модели
                    model_info = self.parser.extract_3d_info_from_text(text)
                    
                    # Сохранение изображения
                    try:
                        image_filename = f"model_{message.id}.jpg"
                        image_path = f"downloaded_images/{image_filename}"
                        await self.parser.client.download_media(message.media, image_path)
                    except Exception as e:
                        logger.warning(f"⚠️ Не удалось сохранить изображение: {e}")
                        image_filename = None
                    
                    # Формирование ссылки на сообщение
                    message_link = f"https://t.me/{TARGET_GROUP_USERNAME.replace('@', '')}/{message.id}"
                    
                    # Сохранение модели в базу
                    self.db.save_model(
                        message_id=message.id,
                        group_id=TARGET_GROUP_USERNAME,
                        post_link=message_link,
                        title=model_info['title'] or f"3D Model {message.id}",
                        description=model_info['description'],
                        image_path=image_filename,
                        embedding=embedding.numpy() if hasattr(embedding, 'numpy') else embedding
                    )
                    
                    new_count += 1
                    logger.info(f"✅ Добавлена новая модель: {model_info['title'][:50]}...")
                    
                except Exception as e:
                    logger.error(f"❌ Ошибка обработки сообщения {message.id}: {e}")
                    continue
            
            # Закрытие соединения
            await self.parser.client.disconnect()
            
            logger.info(f"🎉 Проверка завершена! Добавлено новых моделей: {new_count}")
            return {'new_models': new_count, 'success': True}
            
        except Exception as e:
            logger.error(f"❌ Ошибка проверки новых моделей: {e}")
            return {'error': str(e), 'new_models': 0}

    async def run_continuous_monitoring(self):
        """Непрерывный мониторинг канала"""
        logger.info("🚀 Запуск системы автообновления...")
        logger.info(f"⏱️ Интервал проверки: {self.check_interval // 60} минут")
        
        while True:
            try:
                start_time = datetime.now()
                logger.info(f"🔄 Начинаю проверку в {start_time.strftime('%H:%M:%S')}")
                
                # Проверка новых моделей
                result = await self.check_for_new_models()
                
                end_time = datetime.now()
                duration = (end_time - start_time).seconds
                
                if result.get('success'):
                    new_count = result.get('new_models', 0)
                    total_models = self.db.get_models_count()
                    
                    logger.info(f"✅ Проверка завершена за {duration}с")
                    logger.info(f"📊 Новых моделей: {new_count}")
                    logger.info(f"📊 Всего моделей в базе: {total_models}")
                else:
                    logger.error(f"❌ Ошибка проверки: {result.get('error', 'Unknown')}")
                
                # Ожидание до следующей проверки
                logger.info(f"😴 Следующая проверка через {self.check_interval // 60} минут...")
                await asyncio.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                logger.info("⏹️ Остановка автообновления по запросу пользователя")
                break
            except Exception as e:
                logger.error(f"❌ Критическая ошибка автообновления: {e}")
                logger.info(f"🔄 Перезапуск через 5 минут...")
                await asyncio.sleep(300)  # 5 минут

async def main():
    """Главная функция"""
    updater = AutoUpdater()
    await updater.run_continuous_monitoring()

if __name__ == "__main__":
    asyncio.run(main()) 