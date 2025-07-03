#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import asyncio
import io
from datetime import datetime
from typing import List
import re
import time

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from telegram.constants import ParseMode

from config import BOT_TOKEN, MAX_IMAGE_SIZE, SUPPORTED_FORMATS, MAX_SEARCH_RESULTS
from image_analyzer_lite import ImageAnalyzer
from database_factory import get_database, get_database_info
from group_parser import GroupParser
from live_search import LiveSearchEngine

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class Model3DBot:
    def __init__(self):
        self.image_analyzer = ImageAnalyzer()
        self.database = get_database()
        self.group_parser = GroupParser()
        self.live_search_engine = LiveSearchEngine(self.image_analyzer)
        
        # Статистика использования
        self.usage_stats = {
            'total_searches': 0,
            'successful_searches': 0,
            'total_users': set(),
            'live_searches': 0,
            'db_searches': 0
        }
    
    def _get_category_icon(self, category: str) -> str:
        """Получение иконки для категории"""
        icons = {
            'furniture': '🪑',
            'lighting': '💡',
            'decor': '🏺',
            'architecture': '🏛️',
            'ceiling': '🏠',
            'bathroom': '🚿',
            'kitchen': '🍳',
            'nature': '🌸',
            'vehicle': '🚗',
            'people': '🗿',
            'unknown': '📦'
        }
        return icons.get(category, '📦')
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        user = update.effective_user
        self.usage_stats['total_users'].add(user.id)
        
        welcome_message = f"""
🎨 *Добро пожаловать в 3D Models Search Bot!*

Привет, {user.first_name}! 

Я помогу вам найти похожие 3D-модели по фотографии. Просто отправьте мне изображение предмета мебели или интерьера, и я найду похожие модели в нашей базе данных.

🔍 *Как пользоваться:*
1. Отправьте мне фотографию (.jpg, .png, .webp)
2. Дождитесь анализа изображения
3. Получите ссылки на похожие 3D-модели

📊 *Команды:*
/help - Помощь
/stats - Статистика базы данных
/update - Обновить базу моделей (только для администраторов)

💡 *Совет:* Лучше всего работают четкие фотографии на светлом фоне.

Попробуйте прямо сейчас - отправьте мне любое изображение!
        """
        
        await update.message.reply_text(
            welcome_message,
            parse_mode=ParseMode.MARKDOWN
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /help"""
        help_text = """
🆘 *Справка по использованию*

*Основная функция:*
Отправьте боту фотографию предмета мебели или элемента интерьера, и он найдет похожие 3D-модели.

*Поддерживаемые форматы:*
• JPG/JPEG
• PNG  
• WebP

*Ограничения:*
• Максимальный размер файла: 10 МБ
• Лучше всего работают четкие изображения
• Предмет должен быть хорошо виден на фото

*Команды:*
/start - Начать работу с ботом
/help - Показать эту справку
/stats - Статистика базы данных

*Как получить лучшие результаты:*
1. Делайте фото на светлом фоне
2. Убедитесь, что предмет полностью виден
3. Избегайте размытых изображений
4. Один предмет на фото работает лучше

❓ Если возникли вопросы, свяжитесь с разработчиком.
        """
        
        await update.message.reply_text(
            help_text,
            parse_mode=ParseMode.MARKDOWN
        )
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /stats"""
        db_info = get_database_info()
        models_count = db_info['models_count']
        
        stats_text = f"""
📊 *Статистика бота*

🗄️ База данных: *{db_info['type'].upper()}*
📦 Моделей в базе: *{models_count}*

🔍 *Поисковые запросы:*
• Всего запросов: *{self.usage_stats['total_searches']}*
• Успешных поисков: *{self.usage_stats['successful_searches']}*
• Поиск в БД: *{self.usage_stats['db_searches']}*
• Live поиск: *{self.usage_stats['live_searches']}*

👥 Уникальных пользователей: *{len(self.usage_stats['total_users'])}*

📈 Успешность: *{(self.usage_stats['successful_searches'] / max(1, self.usage_stats['total_searches']) * 100):.1f}%*

⏰ Обновлено: {datetime.now().strftime('%d.%m.%Y %H:%M')}
        """
        
        await update.message.reply_text(
            stats_text,
            parse_mode=ParseMode.MARKDOWN
        )
    
    async def update_database_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /update (только для администраторов)"""
        # Простая проверка администратора (можно расширить)
        admin_ids = [5450018125]  # Ваш Telegram ID
        
        if update.effective_user.id not in admin_ids:
            await update.message.reply_text("❌ У вас нет прав для выполнения этой команды.")
            return
        
        await update.message.reply_text("🔄 Начинаю обновление базы данных...")
        
        try:
            await self.group_parser.update_database(limit=50)
            models_count = self.database.get_models_count()
            
            await update.message.reply_text(
                f"✅ База данных обновлена!\n📊 Текущее количество моделей: {models_count}"
            )
        except Exception as e:
            logger.error(f"Ошибка при обновлении базы: {e}")
            await update.message.reply_text(f"❌ Ошибка при обновлении базы данных: {str(e)}")
    
    async def handle_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик изображений от пользователей"""
        try:
            self.usage_stats['total_searches'] += 1
            self.usage_stats['total_users'].add(update.effective_user.id)
            
            # Проверка наличия изображения
            if not update.message.photo:
                await update.message.reply_text("❌ Пожалуйста, отправьте изображение.")
                return
            
            # Получение самого большого размера фотографии
            photo = update.message.photo[-1]
            
            # Проверка размера файла
            if photo.file_size and photo.file_size > MAX_IMAGE_SIZE:
                await update.message.reply_text(
                    f"❌ Размер файла слишком большой. Максимальный размер: {MAX_IMAGE_SIZE // (1024*1024)} МБ"
                )
                return
            
            # Сохранение информации об изображении для callback
            context.user_data['photo_file_id'] = photo.file_id
            
            # Создание клавиатуры для выбора типа поиска
            models_count = self.database.get_models_count()
            
            # Клавиатура выбора: поиск в базе или глубокий поиск с различным временем
            # При глубоком поиске пользователь сразу выбирает, сколько минут он готов ждать
            keyboard = [
                [
                    InlineKeyboardButton(
                        f"🔍 Поиск в базе ({models_count} моделей)",
                        callback_data="search_db"
                    )
                ],
                [InlineKeyboardButton("🔥 Глубокий поиск – 1 мин", callback_data="search_deep_1")],
                [InlineKeyboardButton("🔥 Глубокий поиск – 3 мин", callback_data="search_deep_3")],
                [InlineKeyboardButton("🔥 Глубокий поиск – 5 мин", callback_data="search_deep_5")],
                [InlineKeyboardButton("🔥 Глубокий поиск – 10 мин", callback_data="search_deep_10")],
                [InlineKeyboardButton("🔥 Глубокий поиск – 30 мин", callback_data="search_deep_30")],
                [InlineKeyboardButton("🔥 Глубокий поиск – 60 мин", callback_data="search_deep_60")],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            message_text = (
                "🔍 *Выберите тип поиска:*\n\n"
                f"🏛️ *База данных:* {models_count} моделей\n"
                "⚡ *Быстро* \\(1\\-3 секунды\\)\n\n"
                "🔥 *Глубокий поиск:* анализ разного объема канала\\.\n"
                "🧠 *Чем больше минут, тем выше точность*\n"
                "⏱️ *Выберите, сколько минут готовы ждать для поиска*"
            )
            await update.message.reply_text(
                message_text,
                reply_markup=reply_markup,
                parse_mode=ParseMode.MARKDOWN_V2
            )
            
        except Exception as e:
            logger.error(f"Ошибка при обработке изображения: {e}")
            await update.message.reply_text(
                "❌ Произошла ошибка при обработке изображения. Попробуйте еще раз."
            )
    
    async def handle_search_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик callback для выбора типа поиска"""
        query = update.callback_query
        await query.answer()
        
        search_type = query.data
        photo_file_id = context.user_data.get('photo_file_id')
        
        if not photo_file_id:
            await query.edit_message_text("❌ Ошибка: изображение не найдено. Отправьте изображение заново.")
            return
        
        try:
            # Скачивание изображения
            file = await context.bot.get_file(photo_file_id)
            image_bytes = await file.download_as_bytearray()
            
            if search_type == "search_db":
                await self.perform_database_search(query, bytes(image_bytes))
            elif search_type.startswith("search_deep_"):
                # Формат: search_deep_<minutes>
                try:
                    minutes = int(search_type.split("_")[-1])
                except ValueError:
                    minutes = 3  # значение по умолчанию
                await self.perform_timed_deep_search(query, bytes(image_bytes), minutes)
            else:
                await query.edit_message_text("❌ Неизвестный тип поиска.")
                
        except Exception as e:
            logger.error(f"Ошибка при выполнении поиска: {e}")
            await query.edit_message_text("❌ Произошла ошибка при поиске. Попробуйте еще раз.")
    
    async def perform_database_search(self, query, image_bytes: bytes):
        """Выполнение поиска в базе данных"""
        await query.edit_message_text("🔍 Поиск в базе данных...")
        
        try:
            self.usage_stats['db_searches'] += 1
            
            # Поиск похожих моделей в базе данных
            similar_models = self.image_analyzer.analyze_and_search(image_bytes)
            
            if not similar_models:
                await query.edit_message_text(
                    "😔 В базе данных не найдено похожих моделей.\n\n"
                    "💡 Попробуйте:\n"
                    "• Загрузить более четкое изображение\n"
                    "• Сфотографировать предмет на светлом фоне"
                )
                return
            
            self.usage_stats['successful_searches'] += 1
            await self.send_search_results_callback(query, similar_models, "база данных")
            
        except Exception as e:
            logger.error(f"Ошибка поиска в базе данных: {e}")
            await query.edit_message_text("❌ Ошибка при поиске в базе данных.")
    
    async def perform_timed_deep_search(self, query, image_bytes: bytes, minutes: int):
        """Глубокий поиск с пользовательским лимитом времени"""
        # Сообщение о начале поиска
        await query.edit_message_text(
            f"🔥 Глубокий поиск в канале @blocks_01 на {minutes} мин…\n"
            "🧠 Анализирую модели для максимального качества\n"
            "⏱️ Пожалуйста, подождите\n\n[░░░░░░░░░░]"
        )

        try:
            self.usage_stats['live_searches'] += 1

            from PIL import Image
            image = Image.open(io.BytesIO(image_bytes))

            # Запускаем кастомный поиск в фоне
            search_task = asyncio.create_task(self.live_search_engine.custom_live_search(image, minutes))

            # Хронометраж для прогресс-бара
            hard_timeout = minutes * 60  # строгий лимит анализа, без запаса
            start_ts = time.time()

            progress_states = [
                "[░░░░░░░░░░]", "[█░░░░░░░░░]", "[██░░░░░░░░]", "[███░░░░░░░]",
                "[████░░░░░░]", "[█████░░░░░]", "[██████░░░░]", "[███████░░░]",
                "[████████░░]", "[█████████░]", "[██████████]"
            ]

            idx = 0
            while not search_task.done() and (time.time() - start_ts) < hard_timeout:
                bar = progress_states[idx % len(progress_states)]
                idx += 1
                await query.edit_message_text(
                    f"🔥 Глубокий поиск в канале @blocks_01 на {minutes} мин…\n"
                    "🧠 Анализирую модели для максимального качества\n"
                    f"{bar}"
                )
                await asyncio.sleep(2.5)

            # Если анализ ещё продолжается (30 мин исчерпаны) – информируем пользователя,
            # далее просто ждём завершения без доп. ограничений.
            if not search_task.done():
                await query.edit_message_text(
                    "⌛ 30-минутный анализ завершён, формирую результаты…"
                )

            # Ждём окончание задачи без дополнительного ограничения времени (отправка быстpая)
            try:
                similar_models = await search_task
            except asyncio.CancelledError:
                # Если задача всё-таки была отменена внутри, возвращаем то, что есть
                similar_models = []
            except Exception as e:
                logger.error(f"Ошибка ожидания результатов глубокого поиска: {e}")
                similar_models = []

            if not similar_models:
                await query.edit_message_text(
                    "😔 Глубокий поиск не нашел похожих моделей.\n\n"
                    "💡 Попробуйте: \n"
                    "• Поиск в базе данных\n"
                    "• Загрузить более четкое изображение"
                )
                return

            self.usage_stats['successful_searches'] += 1
            await self.send_search_results_callback(query, similar_models, f"глубокий поиск {minutes} мин")

        except Exception as e:
            logger.error(f"Ошибка глубокого поиска: {e}")
            await query.edit_message_text("❌ Ошибка при глубоком поиске.")
    
    def _escape_md(self, text: str) -> str:
        """Экранирование спецсимволов для MarkdownV2"""
        if not text:
            return ""
        # Спецсимволы MarkdownV2, которые нужно экранировать
        escape_chars = r"_*[]()~`>#+-=|{}.!"  # '!' тоже
        return re.sub(f"([{re.escape(escape_chars)}])", r"\\\\\\1", text)

    async def send_search_results_callback(self, query, similar_models: List[dict], search_type: str):
        """Отправка результатов поиска для callback"""
        try:
            # Проверяем наличие точных совпадений (visual >= 0.9)
            exact_found = any(m.get('is_exact_match') for m in similar_models)

            if exact_found and search_type.lower().startswith("глуб"):
                header = (
                    f"🎯 Найдены точные совпадения (визуальное сходство ≥ 90%): {len(similar_models)} моделей\n\n"
                )
            elif not exact_found and search_type.lower().startswith("глуб"):
                header = (
                    "😔 Точные совпадения не найдены.\n"
                    f"✅ {search_type.title()}: найдено {len(similar_models)} похожих моделей:\n\n"
                )
            else:
                header = f"✅ {search_type.title()}: найдено {len(similar_models)} моделей:\n\n"

            results_text = self._escape_md(header)
            
            keyboard = []
            
            for i, model in enumerate(similar_models, 1):
                similarity_percent = model.get('similarity', model.get('similarity_score', 0)) * 100
                category = model.get('detected_category', 'unknown')
                category_bonus = model.get('category_bonus', 0.0)
                visual_score = model.get('visual_score', similarity_percent/100)
                
                # Иконка категории
                category_icon = self._get_category_icon(category)
                
                title = self._escape_md(model['title'])
                
                prefix_icon = "🎯" if model.get('is_exact_match') else category_icon
                line = f"*{i}. {prefix_icon} {title}*\n"
                line += f"📊 Сходство: {similarity_percent:.1f}%"
                if category_bonus > 0:
                    line += f" (визуал: {visual_score*100:.1f}% + бонус: {category_bonus*100:.1f}%)"
                line += "\n"
                
                if model.get('description'):
                    desc = model['description'][:80] + "..." if len(model['description']) > 80 else model['description']
                    line += f"📝 {self._escape_md(desc)}\n"
                line += "\n"
                results_text += line
                
                keyboard.append([
                    InlineKeyboardButton(
                        f"🔗 Модель {i} ({similarity_percent:.0f}%)",
                        url=model.get('message_link', model.get('post_link', '#'))
                    )
                ])
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                results_text,
                parse_mode=ParseMode.MARKDOWN_V2,
                reply_markup=reply_markup,
                disable_web_page_preview=True
            )
        except Exception as e:
            logger.error(f"Ошибка при отправке результатов callback: {e}")
            # Пытаемся отправить без форматирования
            try:
                await query.edit_message_text(
                    re.sub(r'[*_`~]', '', results_text),
                    reply_markup=reply_markup
                )
            except Exception as e2:
                logger.error(f"Повторная ошибка отправки результатов: {e2}")
                await query.edit_message_text("❌ Ошибка при отправке результатов.")
    
    async def send_search_results(self, update: Update, processing_message, similar_models: List[dict]):
        """Отправка результатов поиска"""
        try:
            # Удаление сообщения о процессе
            await processing_message.delete()
            
            results_text = f"✅ Найдено {len(similar_models)} похожих моделей:\n\n"
            
            keyboard = []
            
            for i, model in enumerate(similar_models, 1):
                similarity_percent = model['similarity_score'] * 100
                
                results_text += f"*{i}. {model['title']}*\n"
                results_text += f"📊 Сходство: {similarity_percent:.1f}%\n"
                
                if model['description']:
                    # Ограничиваем описание
                    desc = model['description'][:100] + "..." if len(model['description']) > 100 else model['description']
                    results_text += f"📝 {desc}\n"
                
                results_text += "\n"
                
                # Добавляем кнопку для каждой модели
                keyboard.append([
                    InlineKeyboardButton(
                        f"🔗 Модель {i} ({similarity_percent:.0f}%)",
                        url=model['post_link']
                    )
                ])
            
            # Кнопка для повторного поиска
            keyboard.append([
                InlineKeyboardButton("🔄 Попробовать другое изображение", callback_data="search_again")
            ])
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                results_text,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=reply_markup,
                disable_web_page_preview=True
            )
            
        except Exception as e:
            logger.error(f"Ошибка при отправке результатов: {e}")
            await update.message.reply_text("❌ Ошибка при отправке результатов.")
    
    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик текстовых сообщений"""
        await update.message.reply_text(
            "📷 Пожалуйста, отправьте изображение для поиска похожих 3D-моделей.\n\n"
            "Поддерживаемые форматы: JPG, PNG, WebP"
        )
    
    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик ошибок"""
        logger.error(f"Ошибка: {context.error}")
        
        if update and update.message:
            await update.message.reply_text(
                "❌ Произошла внутренняя ошибка. Попробуйте позже."
            )

def main():
    """Основная функция запуска бота"""
    print("🚀 Запуск 3D Models Search Bot...")
    
    # Создание экземпляра бота
    bot = Model3DBot()
    
    # Создание приложения
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Добавление обработчиков
    application.add_handler(CommandHandler("start", bot.start_command))
    application.add_handler(CommandHandler("help", bot.help_command))
    application.add_handler(CommandHandler("stats", bot.stats_command))
    application.add_handler(CommandHandler("update", bot.update_database_command))
    
    # Обработчик callback для выбора типа поиска
    application.add_handler(CallbackQueryHandler(bot.handle_search_callback))
    
    # Обработчик изображений
    application.add_handler(MessageHandler(filters.PHOTO, bot.handle_image))
    
    # Обработчик текста
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_text))
    
    # Обработчик ошибок
    application.add_error_handler(bot.error_handler)
    
    print("✅ Бот запущен и готов к работе!")
    print(f"📊 Моделей в базе данных: {bot.database.get_models_count()}")
    
    # Запуск бота
    application.run_polling()

if __name__ == '__main__':
    main() 