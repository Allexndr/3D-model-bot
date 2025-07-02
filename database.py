import sqlite3
import json
import numpy as np
from typing import List, Tuple, Optional
from config import DATABASE_PATH

class ModelsDatabase:
    def __init__(self):
        self.db_path = DATABASE_PATH
        self.init_database()
    
    def init_database(self):
        """Инициализация базы данных"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Таблица для хранения информации о моделях
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id INTEGER UNIQUE,
                    group_id TEXT,
                    post_link TEXT,
                    title TEXT,
                    description TEXT,
                    image_path TEXT,
                    embedding BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Таблица для метаданных группы
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS group_metadata (
                    group_id TEXT PRIMARY KEY,
                    last_parsed_message_id INTEGER,
                    total_models INTEGER DEFAULT 0,
                    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def save_model(self, message_id: int, group_id: str, post_link: str, 
                   title: str, description: str, image_path: str, embedding: np.ndarray):
        """Сохранение модели в базу данных"""
        embedding_blob = embedding.tobytes()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO models 
                (message_id, group_id, post_link, title, description, image_path, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (message_id, group_id, post_link, title, description, image_path, embedding_blob))
            
            conn.commit()
    
    def get_all_embeddings(self) -> List[Tuple[int, np.ndarray]]:
        """Получение всех эмбеддингов для поиска"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT id, embedding FROM models")
            results = cursor.fetchall()
            
            embeddings = []
            for model_id, embedding_blob in results:
                embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                embeddings.append((model_id, embedding))
            
            return embeddings
    
    def get_model_by_id(self, model_id: int) -> Optional[dict]:
        """Получение информации о модели по ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT message_id, group_id, post_link, title, description, image_path
                FROM models WHERE id = ?
            """, (model_id,))
            
            result = cursor.fetchone()
            if result:
                return {
                    'message_id': result[0],
                    'group_id': result[1],
                    'post_link': result[2],
                    'title': result[3],
                    'description': result[4],
                    'image_path': result[5]
                }
            return None
    
    def update_group_metadata(self, group_id: str, last_message_id: int):
        """Обновление метаданных группы"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO group_metadata 
                (group_id, last_parsed_message_id, last_update)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """, (group_id, last_message_id))
            
            conn.commit()
    
    def get_group_metadata(self, group_id: str) -> Optional[dict]:
        """Получение метаданных группы"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT last_parsed_message_id, total_models, last_update
                FROM group_metadata WHERE group_id = ?
            """, (group_id,))
            
            result = cursor.fetchone()
            if result:
                return {
                    'last_parsed_message_id': result[0],
                    'total_models': result[1],
                    'last_update': result[2]
                }
            return None
    
    def get_models_count(self) -> int:
        """Получение общего количества моделей"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM models")
            return cursor.fetchone()[0]
    
    def get_model_by_message_id(self, message_id: int) -> Optional[dict]:
        """Получение модели по ID сообщения Telegram"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, message_id, group_id, post_link, title, description, image_path
                FROM models WHERE message_id = ?
            """, (message_id,))
            
            result = cursor.fetchone()
            if result:
                return {
                    'id': result[0],
                    'message_id': result[1],
                    'group_id': result[2],
                    'post_link': result[3],
                    'title': result[4],
                    'description': result[5],
                    'image_path': result[6]
                }
            return None

    def get_max_message_id(self) -> int:
        """Получить максимальный message_id из базы данных"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT MAX(message_id) FROM models")
                result = cursor.fetchone()
                return result[0] if result[0] is not None else 0
        except Exception as e:
            print(f"❌ Ошибка получения максимального message_id: {e}")
            return 0 