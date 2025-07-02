#!/usr/bin/env python3
"""
Live поиск 3D моделей без сохранения в базу данных
Ищет похожие модели в реальном времени прямо в канале
"""

import asyncio
import logging
import time
from typing import List, Dict, Optional, Tuple
from telethon import TelegramClient, errors
from telethon.tl.types import MessageMediaPhoto, MessageMediaDocument
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import torch
from io import BytesIO

from config import TELEGRAM_API_ID as API_ID, TELEGRAM_API_HASH as API_HASH, TARGET_GROUP_USERNAME
from image_analyzer import ImageAnalyzer

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LiveSearchEngine:
    def __init__(self, image_analyzer: ImageAnalyzer | None = None):
        """Создаёт движок поиска.

        Args:
            image_analyzer: Уже инициализированный ImageAnalyzer. Если не
                передан, будет создан новый экземпляр. Передавая готовый
                экземпляр из бота, мы избегаем повторной загрузки CLIP-модели,
                экономя 1-2 секунды запуска и ~2× оперативной памяти.
        """
        self.client = None
        self.image_analyzer = image_analyzer or ImageAnalyzer()
        self.session_file = '3d_parser_session.session'
    
    async def authenticate(self) -> bool:
        """Аутентификация в Telegram"""
        try:
            self.client = TelegramClient(self.session_file, API_ID, API_HASH)
            await self.client.start()
            
            me = await self.client.get_me()
            logger.info(f"✅ Аутентификация успешна: {me.first_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка аутентификации: {e}")
            return False
    
    async def download_and_analyze_image(self, message) -> Optional[np.ndarray]:
        """Скачивание и анализ изображения из сообщения"""
        try:
            # Скачивание превью (thumbnail) вместо полного файла, чтобы
            # сэкономить трафик и время. Если превью нет – скачиваем оригинал.
            image_bytes = BytesIO()

            try:
                # thumb=-1 – самый большой thumbnail (обычно ≤ 320 px)
                await self.client.download_media(message.media, image_bytes, thumb=-1)
                if image_bytes.getbuffer().nbytes == 0:
                    raise ValueError("Empty thumbnail")
            except Exception:
                # Фолбэк: скачиваем оригинальный файл
                image_bytes.seek(0)
                image_bytes.truncate(0)
                await self.client.download_media(message.media, image_bytes)
            
            # Анализ изображения
            image_bytes.seek(0)
            image = Image.open(image_bytes)
            
            # Получение эмбеддинга
            embedding = self.image_analyzer.process_image(image)
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа изображения: {e}")
            return None
    
    def extract_model_info(self, text: str, message_id: int) -> Dict[str, str]:
        """Извлечение информации о 3D модели из текста"""
        title = f"3D Model {message_id}"
        description = text[:200] if text else "Архитектурная 3D модель"
        
        # Простое извлечение заголовка из первой строки
        if text:
            lines = text.split('\n')
            if lines and len(lines[0]) > 5:
                title = lines[0][:50]
        
        return {
            'title': title,
            'description': description
        }
    
    async def live_search_similar_models(self, 
                                       query_image: Image.Image, 
                                       max_results: int = 5,
                                       search_limit: Optional[int] = 2000,
                                       exact_threshold: float = 0.90,
                                       time_limit_seconds: Optional[int] = None) -> List[Dict]:
        """
        Live поиск похожих моделей в канале в реальном времени
        
        Args:
            query_image: Изображение для поиска
            max_results: Максимум результатов
            search_limit: Сколько последних сообщений искать. Если None, ищем по всей истории.
            exact_threshold: Порог визуального сходства (0-1) для точного совпадения.
            time_limit_seconds: Лимит времени для анализа в секундах. Если None, лимит не используется.
        """
        try:
            logger.info(f"🔍 Начинаю качественный live поиск в канале {TARGET_GROUP_USERNAME}")
            if search_limit is None:
                logger.info("📊 Будем искать во всей истории канала для максимального качества")
            else:
                logger.info(f"📊 Будем искать в {search_limit} последних сообщениях")
            
            # Аутентификация
            if not await self.authenticate():
                return []
            
            # Анализ входящего изображения
            logger.info("🖼️ Анализирую входящее изображение...")
            query_embedding = self.image_analyzer.process_image(query_image)
            if query_embedding is None:
                logger.error("❌ Не удалось проанализировать входящее изображение")
                return []
            
            # Определение категории запроса
            query_category = self.image_analyzer.detect_query_category(query_image)
            logger.info(f"🎯 Категория запроса: {query_category}")
            
            # Список найденных моделей по категориям
            exact_visual_matches = []  # Точные совпадения по визуалу (>= exact_threshold)
            category_matches = []      # Точные совпадения категорий
            related_matches = []   # Родственные категории  
            other_matches = []     # Остальные результаты
            processed_count = 0
            
            # Поиск в канале с увеличенным лимитом
            start_ts = time.time()
            async for message in self.client.iter_messages(TARGET_GROUP_USERNAME, limit=search_limit):
                try:
                    # Проверка тайм-аута
                    if time_limit_seconds is not None and (time.time() - start_ts) > time_limit_seconds:
                        logger.info("⏰ Достигли лимита времени анализа, завершаем перебор сообщений…")
                        break
                    
                    # Проверка наличия медиа
                    if not message.media:
                        continue
                    
                    # Проверка типа медиа (только изображения)
                    is_image = False
                    if hasattr(message.media, 'photo'):
                        is_image = True
                    elif hasattr(message.media, 'document'):
                        if message.media.document.mime_type and message.media.document.mime_type.startswith('image/'):
                            is_image = True
                    
                    if not is_image:
                        continue
                    
                    # Проверка на 3D модель по тексту
                    text = message.message or ""
                    text_lower = text.lower()
                    model_keywords = ['3d', 'модель', 'model', 'max', 'obj', 'fbx', 'архитектура', 'мебель', 'decor', 'interior', 'exterior', 'furniture']
                    
                    is_3d_model = any(keyword in text_lower for keyword in model_keywords) or len(text) < 20
                    
                    if not is_3d_model:
                        continue
                    
                    processed_count += 1
                    if processed_count % 10 == 0:
                        logger.info(f"🔍 Проанализировано {processed_count} моделей...")
                    
                    # Анализ изображения модели
                    model_embedding = await self.download_and_analyze_image(message)
                    if model_embedding is None:
                        continue
                    
                    # Вычисление визуального сходства
                    visual_score = cosine_similarity(
                        query_embedding.reshape(1, -1),
                        model_embedding.reshape(1, -1)
                    )[0][0]
                    
                    # Проверка на точное визуальное совпадение (без бонусов)
                    if visual_score >= exact_threshold:
                        # Извлечение информации о модели
                        model_info = self.extract_model_info(text, message.id)

                        message_link = f"https://t.me/{TARGET_GROUP_USERNAME.replace('@', '')}/{message.id}"

                        exact_visual_matches.append({
                            'title': model_info['title'],
                            'description': model_info['description'],
                            'similarity': float(visual_score),
                            'visual_score': float(visual_score),
                            'category_bonus': 0.0,
                            'detected_category': 'unknown',
                            'category_match_type': 'exact_visual',
                            'is_exact_match': True,
                            'message_link': message_link,
                            'message_id': message.id
                        })
                        logger.info(f"🎯 Найдено точное визуальное совпадение {visual_score:.3f} - {model_info['title'][:40]}...")

                        # Собрали достаточно точных совпадений
                        if len(exact_visual_matches) >= max_results:
                            break

                        # Продолжаем поиск других точных совпадений без применения порога 0.45
                        continue  # Переходим к следующему сообщению

                    # Фильтр по минимальному сходству для общего поиска
                    if visual_score < 0.45:
                        continue
                    
                    # Извлечение информации о модели
                    model_info = self.extract_model_info(text, message.id)
                    
                    # Определение категории найденной модели
                    model_category = self.image_analyzer.classify_image_by_text(text)
                    
                    # Усиленная категориальная система бонусов
                    category_bonus = 0.0
                    category_match_type = 'none'
                    
                    if query_category != 'unknown' and model_category != 'unknown':
                        if query_category == model_category:
                            category_bonus = 0.25  # Увеличен бонус за точное совпадение
                            category_match_type = 'exact'
                        elif self.image_analyzer._categories_related(query_category, model_category):
                            category_bonus = 0.12  # Увеличен бонус за родственные категории
                            category_match_type = 'related'
                    
                    # Итоговый скор
                    final_score = visual_score + category_bonus
                    
                    # Формирование ссылки
                    message_link = f"https://t.me/{TARGET_GROUP_USERNAME.replace('@', '')}/{message.id}"
                    
                    result = {
                        'title': model_info['title'],
                        'description': model_info['description'],
                        'similarity': float(final_score),
                        'visual_score': float(visual_score),
                        'category_bonus': float(category_bonus),
                        'detected_category': model_category,
                        'category_match_type': category_match_type,
                        'is_exact_match': False,
                        'message_link': message_link,
                        'message_id': message.id
                    }
                    
                    # Распределение по категориям для приоритизации
                    if category_match_type == 'exact':
                        category_matches.append(result)
                        logger.info(f"🎯 Точное совпадение: {final_score:.3f} ({model_category}) - {model_info['title'][:30]}...")
                    elif category_match_type == 'related':
                        related_matches.append(result)
                        logger.info(f"🔗 Родственная категория: {final_score:.3f} ({model_category}) - {model_info['title'][:30]}...")
                    else:
                        other_matches.append(result)
                        logger.info(f"📦 Другая категория: {final_score:.3f} ({model_category}) - {model_info['title'][:30]}...")
                    
                except Exception as e:
                    logger.error(f"❌ Ошибка обработки сообщения {message.id}: {e}")
                    continue
            
            # Закрытие соединения
            await self.client.disconnect()
            
            # Приоритетная сортировка результатов
            logger.info(f"📊 Категориальное распределение найденных моделей:")
            logger.info(f"   🎯 Точных совпадений ({query_category}): {len(category_matches)}")
            logger.info(f"   🔗 Родственных категорий: {len(related_matches)}")
            logger.info(f"   📦 Других категорий: {len(other_matches)}")
            
            # Сортировка каждой группы
            category_matches.sort(key=lambda x: x['similarity'], reverse=True)
            related_matches.sort(key=lambda x: x['similarity'], reverse=True)
            other_matches.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Если есть точные визуальные совпадения – возвращаем только их
            if exact_visual_matches:
                exact_visual_matches.sort(key=lambda x: x['similarity'], reverse=True)
                final_results = exact_visual_matches[:max_results]
                logger.info(f"🏆 Найдено {len(final_results)} точных визуальных совпадений. Завершаем поиск.")
                # Закрытие соединения
                await self.client.disconnect()
                return final_results

            # Приоритетное объединение результатов (если точных совпадений нет)
            final_results = []
            if query_category != 'unknown':
                if len(category_matches) >= max_results:
                    final_results = category_matches[:max_results]
                elif len(category_matches) > 0:
                    remaining_slots = max_results - len(category_matches)
                    final_results = category_matches[:]
                    final_results.extend(related_matches[:remaining_slots//2])
                    final_results.extend(other_matches[:remaining_slots - len(related_matches[:remaining_slots//2])])
                    final_results = final_results[:max_results]
                else:
                    final_results = related_matches[:max_results//2] + other_matches[:max_results//2]
            else:
                all_results = category_matches + related_matches + other_matches
                all_results.sort(key=lambda x: x['similarity'], reverse=True)
                final_results = all_results[:max_results]
            
            logger.info(f"🎉 Live поиск завершен!")
            logger.info(f"📊 Обработано сообщений: {processed_count}")
            logger.info(f"🔍 Найдено релевантных моделей: {len(category_matches + related_matches + other_matches)}")
            logger.info(f"🏆 Итоговых результатов: {len(final_results)}")
            
            # Вывод итоговых результатов
            for i, result in enumerate(final_results, 1):
                match_icon = "🎯" if result['category_match_type'] == 'exact' else "🔗" if result['category_match_type'] == 'related' else "📦"
                logger.info(f"   {i}. {match_icon} {result['similarity']:.3f} ({result['detected_category']}) - {result['title'][:40]}...")
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Ошибка live поиска: {e}")
            return []
    
    async def quick_live_search(self, query_image: Image.Image) -> List[Dict]:
        """Быстрый live поиск (500 последних сообщений)"""
        return await self.live_search_similar_models(
            query_image=query_image,
            max_results=5,
            search_limit=500  # Увеличено для лучшего качества
        )
    
    async def deep_live_search(self, query_image: Image.Image) -> List[Dict]:
        """Глубокий live поиск по всей истории канала с приоритетом точного совпадения"""
        return await self.live_search_similar_models(
            query_image=query_image,
            max_results=8,
            search_limit=None,  # Ищем по всей истории
            exact_threshold=0.90
        )

    async def custom_live_search(self, query_image: Image.Image, minutes: int = 3, max_results: int = 8) -> List[Dict]:
        """Пользовательский live-поиск с ограничением по выделенному времени (в минутах).

        Подбирает приблизительный search_limit (кол-во сообщений для анализа) исходя из
        указанного пользователем времени поиска. Цель — линейно масштабировать глубину
        истории канала, сохраняя приемлемое время ответа.

        Args:
            query_image: Изображение-запрос.
            minutes: Сколько минут пользователь готов ждать (1-60).
            max_results: Сколько результатов вернуть.
        """
        # Безопасные границы
        minutes = max(1, min(minutes, 60))

        # До 30 минут – анализируем minutes*500 последних сообщений.  
        # 30 минут и более – просматриваем всю историю канала.

        messages_per_min = 500

        if minutes >= 30:
            search_limit = None  # вся история
        else:
            search_limit = minutes * messages_per_min

        # Для очень коротких промежутков (<2 мин) уменьшаем max_results для скорости
        if minutes <= 2 and max_results > 5:
            max_results = 5

        return await self.live_search_similar_models(
            query_image=query_image,
            max_results=max_results,
            search_limit=search_limit,
            exact_threshold=0.90,
            time_limit_seconds=minutes*60,
        )

# Тестирование live поиска
async def test_live_search():
    """Тестирование live поиска"""
    from PIL import Image
    
    # Создаем тестовое изображение
    test_image = Image.new('RGB', (256, 256), color='white')
    
    search_engine = LiveSearchEngine()
    
    print("🧪 Тестирование Live Search...")
    print("=" * 50)
    
    # Быстрый поиск
    print("⚡ Быстрый live поиск (500 сообщений)...")
    start_time = time.time()
    results = await search_engine.quick_live_search(test_image)
    quick_time = time.time() - start_time
    
    print(f"⏱️ Время быстрого поиска: {quick_time:.1f} секунд")
    print(f"📊 Найдено результатов: {len(results)}")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['similarity']:.3f} - {result['title']}")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    asyncio.run(test_live_search()) 