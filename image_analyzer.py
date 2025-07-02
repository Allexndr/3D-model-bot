import torch
import torch.nn.functional as F
import os

# --- Пытаемся использовать более мощную OpenCLIP ---
try:
    import open_clip
    _OPEN_CLIP_AVAILABLE = True
except ImportError:
    _OPEN_CLIP_AVAILABLE = False

# Huggingface CLIP как запасной вариант
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from typing import List, Tuple, Dict
from sklearn.metrics.pairwise import cosine_similarity
import io
import requests
import re

from config import CLIP_MODEL_NAME, IMAGE_SIZE, MAX_SEARCH_RESULTS
from database_factory import get_database

class ImageAnalyzer:
    def __init__(self):
        print("Загрузка CLIP-модели высокого качества…")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.use_open_clip = False

        # Попытка загрузить OpenCLIP ViT-L/14 (LAION-2B) – более высокая точность
        if _OPEN_CLIP_AVAILABLE:
            try:
                cache_dir = os.path.expanduser("~/.cache/open_clip")
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    "ViT-L-14", pretrained="laion2b_s32b_b82k", cache_dir=cache_dir
                )
                self.tokenizer = open_clip.get_tokenizer("ViT-L-14")
                self.model.to(self.device)
                self.model.eval()
                self.use_open_clip = True
                print("✅ Используется OpenCLIP ViT-L/14 (LAION-2B)")
            except Exception as e:
                print(f"⚠️ Не удалось загрузить OpenCLIP: {e}. Переключаемся на transformers.")

        if not self.use_open_clip:
            # Fallback: обычный CLIP через transformers (модель задана в config.py)
            self.model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
            self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Используется модель {CLIP_MODEL_NAME} из transformers")
        
        self.database = get_database()
        print(f"CLIP модель загружена на {self.device}")
        
        # Категории для улучшенного поиска
        self.categories = {
            'furniture': {
                'keywords': ['стул', 'chair', 'кресло', 'armchair', 'диван', 'sofa', 'стол', 'table', 'шкаф', 'wardrobe', 'мебель', 'furniture', 'полка', 'shelf'],
                'weight': 1.2
            },
            'lighting': {
                'keywords': ['лампа', 'lamp', 'светильник', 'light', 'люстра', 'chandelier', 'бра', 'sconce'],
                'weight': 1.1
            },
            'decor': {
                'keywords': ['декор', 'decor', 'ваза', 'vase', 'картина', 'picture', 'статуэтка', 'figurine'],
                'weight': 1.1
            },
            'architecture': {
                'keywords': ['архитектура', 'architecture', 'здание', 'building', 'фасад', 'facade', 'окно', 'window', 'дверь', 'door'],
                'weight': 1.0
            },
            'ceiling': {
                'keywords': ['потолок', 'ceiling', 'wave', 'волна', 'покрытие', 'coating'],
                'weight': 1.0
            },
            'bathroom': {
                'keywords': ['ванна', 'bath', 'душ', 'shower', 'унитаз', 'toilet', 'раковина', 'sink', 'зеркало', 'mirror'],
                'weight': 1.0
            },
            'kitchen': {
                'keywords': ['кухня', 'kitchen', 'плита', 'stove', 'холодильник', 'fridge', 'техника', 'appliance'],
                'weight': 1.0
            },
            'nature': {
                'keywords': ['природа', 'nature', 'дерево', 'tree', 'цветы', 'flowers', 'животное', 'animal'],
                'weight': 1.0
            },
            'other': {
                'keywords': ['другое', 'other', 'неизвестное', 'unknown'],
                'weight': 1.0
            }
        }
    
    def classify_image_by_text(self, text: str) -> str:
        """Классификация изображения по текстовому описанию"""
        if not text:
            return 'unknown'
        
        text_lower = text.lower()
        
        for category, data in self.categories.items():
            if any(keyword in text_lower for keyword in data['keywords']):
                return category
        
        return 'unknown'
    
    def detect_query_category(self, image: Image.Image) -> str:
        """Определение категории входящего изображения через CLIP с порогом уверенности"""
        try:
            category_prompts = [
                "a piece of furniture (chair/sofa/table)",
                "a lighting object (lamp/chandelier)",
                "a decorative object (vase/statue/picture)",
                "an architectural element (building/facade)",
                "a ceiling design element",
                "a bathroom item (mirror/sink/bathtub)",
                "a kitchen appliance (stove/fridge)",
                "a natural object (flower/plant/tree)",
                "a vehicle (car/truck)",
                "a human/character statue",
            ]
            category_names = [
                'furniture', 'lighting', 'decor', 'architecture', 'ceiling',
                'bathroom', 'kitchen', 'nature', 'vehicle', 'people'
            ]
            
            if self.use_open_clip:
                # Получаем эмбеддинг изображения
                img_tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    img_feat = F.normalize(self.model.encode_image(img_tensor), p=2, dim=1)

                # Эмбеддинги текстовых промптов
                text_tokens = self.tokenizer(category_prompts).to(self.device)
                with torch.no_grad():
                    txt_feat = F.normalize(self.model.encode_text(text_tokens), p=2, dim=1)

                sims = (img_feat @ txt_feat.T).squeeze(0)
            else:
                image_inputs = self.processor(images=image, return_tensors="pt")
                text_inputs = self.processor(text=category_prompts, return_tensors="pt", padding=True, truncation=True)
                image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
                text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

                with torch.no_grad():
                    img_feat = F.normalize(self.model.get_image_features(**image_inputs), p=2, dim=1)
                    txt_feat = F.normalize(self.model.get_text_features(**text_inputs), p=2, dim=1)
                    sims = torch.cosine_similarity(img_feat, txt_feat, dim=1)
            best_idx = torch.argmax(sims).item()
            best_category = category_names[best_idx]
            confidence = sims[best_idx].item()
            print(f"🎯 Категория: {best_category} | уверенность: {confidence:.3f}")
            
            # Минимальный порог уверенности
            if confidence < 0.15:
                return 'unknown'
            return best_category
        except Exception as e:
            print(f"Ошибка при определении категории: {e}")
            return 'unknown'

    def process_image(self, image: Image.Image) -> np.ndarray:
        """Обработка изображения и получение эмбеддинга"""
        try:
            if self.use_open_clip:
                # OpenCLIP путь
                image = image.convert("RGB")
                image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    image_features = self.model.encode_image(image_tensor)
                    image_features = F.normalize(image_features, p=2, dim=1)
                return image_features.cpu().numpy().flatten()
            else:
                # Huggingface CLIP путь
                image = image.convert('RGB')
                image = image.resize(IMAGE_SIZE, Image.LANCZOS)

                inputs = self.processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs)
                    image_features = F.normalize(image_features, p=2, dim=1)

                return image_features.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"Ошибка при обработке изображения: {e}")
            return None
    
    def process_image_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Обработка изображения из байтов"""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            return self.process_image(image)
        except Exception as e:
            print(f"Ошибка при открытии изображения из байтов: {e}")
            return None
    
    def process_image_from_url(self, image_url: str) -> np.ndarray:
        """Обработка изображения по URL"""
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            return self.process_image_from_bytes(response.content)
        except Exception as e:
            print(f"Ошибка при загрузке изображения по URL {image_url}: {e}")
            return None
    
    def find_similar_models_enhanced(self, query_embedding: np.ndarray, query_category: str = 'unknown', top_k: int = MAX_SEARCH_RESULTS) -> List[dict]:
        """Улучшенный поиск похожих моделей с усиленной категориальной фильтрацией"""
        try:
            # Получение всех моделей из базы данных
            all_models = self.database.get_all_models()
            stored_embeddings = self.database.get_all_embeddings()
            
            if not stored_embeddings or not all_models:
                print("База данных моделей пуста")
                return []
            
            # Создание словаря для быстрого доступа к модели по ID
            models_dict = {model['message_id']: model for model in all_models}
            
            # Подготовка данных для поиска
            model_ids = [item[0] for item in stored_embeddings]
            embeddings_matrix = np.array([item[1] for item in stored_embeddings])
            
            # Нормализация эмбеддингов
            query_embedding = query_embedding.reshape(1, -1)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            embeddings_matrix = embeddings_matrix / np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
            
            # Вычисление визуального сходства
            visual_similarities = cosine_similarity(query_embedding, embeddings_matrix).flatten()
            
            # Подготовка результатов с усиленной категориальной фильтрацией
            enhanced_results = []
            category_matches = []  # Точные совпадения категорий
            related_matches = []   # Родственные категории
            other_matches = []     # Остальные результаты
            
            for idx, model_id in enumerate(model_ids):
                if model_id not in models_dict:
                    continue
                    
                model_info = models_dict[model_id].copy()
                visual_score = visual_similarities[idx]
                
                # Фильтр по минимальному визуальному сходству
                if visual_score < 0.45:  # Повышен порог для качества
                    continue
                
                # Определение категории модели
                model_text = (model_info.get('title', '') + ' ' + model_info.get('description', '')).lower()
                model_category = self.classify_image_by_text(model_text)
                
                # Усиленная категориальная система бонусов
                category_bonus = 0.0
                category_match_type = 'none'
                
                if query_category != 'unknown' and model_category != 'unknown':
                    if query_category == model_category:
                        category_bonus = 0.25  # Увеличен бонус за точное совпадение
                        category_match_type = 'exact'
                    elif self._categories_related(query_category, model_category):
                        category_bonus = 0.12  # Увеличен бонус за родственные категории
                        category_match_type = 'related'
                
                # Итоговый скор с учетом категории
                final_score = min(1.0, visual_score + category_bonus)
                
                model_info['similarity_score'] = float(final_score)
                model_info['visual_score'] = float(visual_score)
                model_info['category_bonus'] = float(category_bonus)
                model_info['detected_category'] = model_category
                model_info['category_match_type'] = category_match_type
                
                # Распределение по категориям для приоритизации
                if category_match_type == 'exact':
                    category_matches.append(model_info)
                elif category_match_type == 'related':
                    related_matches.append(model_info)
                else:
                    other_matches.append(model_info)
            
            # Сортировка каждой группы по скору
            category_matches.sort(key=lambda x: x['similarity_score'], reverse=True)
            related_matches.sort(key=lambda x: x['similarity_score'], reverse=True)
            other_matches.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            # --- Fallback: если результатов слишком мало, повторяем с пониженным порогом ---
            if len(category_matches) + len(related_matches) + len(other_matches) == 0:
                print("⚠️ Нет результатов при пороге 0.45 — снижаем порог до 0.35")
                min_visual = 0.35
            else:
                min_visual = None
            
            if min_visual:
                for idx, model_id in enumerate(model_ids):
                    if model_id not in models_dict:
                        continue
                    model_info = models_dict[model_id].copy()
                    visual_score = visual_similarities[idx]
                    if visual_score < min_visual:
                        continue
                    model_info['similarity_score'] = float(visual_score)
                    other_matches.append(model_info)
            
            # --- Пересчитываем, если после фильтра мало category_matches ---
            if query_category != 'unknown' and len(category_matches) < 2:
                print("⚠️ Слишком мало точных совпадений категории — отключаем категориальные бонусы")
                query_category = 'unknown'
            
            # Приоритетное объединение результатов
            if query_category != 'unknown':
                print(f"🎯 Категориальная фильтрация:")
                print(f"   ✅ Точных совпадений ({query_category}): {len(category_matches)}")
                print(f"   🔗 Родственных категорий: {len(related_matches)}")
                print(f"   📦 Других категорий: {len(other_matches)}")
                
                # Если есть точные совпадения категории, приоритизируем их
                if len(category_matches) >= top_k:
                    enhanced_results = category_matches[:top_k]
                elif len(category_matches) > 0:
                    # Берем все точные + лучшие родственные/другие
                    remaining_slots = top_k - len(category_matches)
                    enhanced_results = category_matches[:]
                    enhanced_results.extend(related_matches[:remaining_slots//2])
                    enhanced_results.extend(other_matches[:remaining_slots - len(related_matches[:remaining_slots//2])])
                    enhanced_results = enhanced_results[:top_k]
                else:
                    # Нет точных совпадений, берем родственные + другие
                    enhanced_results = related_matches[:top_k//2] + other_matches[:top_k//2]
            else:
                # Без категории - обычная сортировка
                all_results = category_matches + related_matches + other_matches
                all_results.sort(key=lambda x: x['similarity_score'], reverse=True)
                enhanced_results = all_results[:top_k]
            
            print(f"🔍 Итоговые результаты ({len(enhanced_results)}):")
            for i, result in enumerate(enhanced_results):
                match_type_icon = "🎯" if result['category_match_type'] == 'exact' else "🔗" if result['category_match_type'] == 'related' else "📦"
                print(f"  {i+1}. {match_type_icon} {result['title'][:35]}... (скор: {result['similarity_score']:.3f}, виз: {result['visual_score']:.3f}, кат: {result['detected_category']})")
            
            return enhanced_results
            
        except Exception as e:
            print(f"Ошибка при улучшенном поиске похожих моделей: {e}")
            return []
    
    def _categories_related(self, cat1: str, cat2: str) -> bool:
        """Проверка родственности категорий"""
        related_groups = [
            {'furniture', 'lighting', 'decor'},  # Интерьерные объекты
            {'bathroom', 'kitchen'},  # Помещения с сантехникой
            {'architecture', 'ceiling'}  # Архитектурные элементы
        ]
        
        for group in related_groups:
            if cat1 in group and cat2 in group:
                return True
        return False

    def find_similar_models(self, query_embedding: np.ndarray, top_k: int = MAX_SEARCH_RESULTS) -> List[dict]:
        """Обратная совместимость - используем улучшенный поиск"""
        return self.find_similar_models_enhanced(query_embedding, 'unknown', top_k)
    
    def analyze_and_search(self, image_bytes: bytes) -> List[dict]:
        """Полный цикл анализа изображения и поиска с улучшенным алгоритмом"""
        print("🔍 Анализ изображения...")
        
        # Открываем изображение для категоризации
        image = Image.open(io.BytesIO(image_bytes))
        
        # Определяем категорию изображения
        query_category = self.detect_query_category(image)
        
        # Получение эмбеддинга изображения
        embedding = self.process_image_from_bytes(image_bytes)
        if embedding is None:
            return []
        
        print(f"🎯 Поиск похожих моделей в категории: {query_category}")
        
        # Улучшенный поиск с учетом категории
        similar_models = self.find_similar_models_enhanced(embedding, query_category)
        
        return similar_models
    
    def analyze_image(self, image_path: str) -> np.ndarray:
        """Анализ изображения по пути к файлу"""
        try:
            image = Image.open(image_path)
            return self.process_image(image)
        except Exception as e:
            print(f"Ошибка при анализе изображения {image_path}: {e}")
            return None

    def get_text_embedding(self, text: str) -> np.ndarray:
        """Получение эмбеддинга текста (для описаний моделей)"""
        try:
            inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                text_features = F.normalize(text_features, p=2, dim=1)
                
            return text_features.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"Ошибка при обработке текста: {e}")
            return None
    
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
            'nature': '🌳',
            'other': '📦',
            'unknown': '📦'
        }
        return icons.get(category, '📦') 