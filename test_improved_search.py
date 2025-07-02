#!/usr/bin/env python3
"""
Тестирование улучшенного алгоритма поиска с категориальной фильтрацией
"""

import asyncio
from PIL import Image
import numpy as np
from image_analyzer import ImageAnalyzer
from database_factory import get_database

def create_test_image(size=(224, 224), color='RGB'):
    """Создание тестового изображения"""
    return Image.new(color, size, (255, 255, 255))

async def test_improved_search():
    """Тестирование улучшенного поиска"""
    print("🧪 Тестирование улучшенного алгоритма поиска")
    print("=" * 60)
    
    # Инициализация
    analyzer = ImageAnalyzer()
    database = get_database()
    
    # Статистика базы данных
    all_models = database.get_all_models()
    print(f"📊 Моделей в базе данных: {len(all_models)}")
    
    # Анализ категорий в базе
    print("\n📈 Анализ категорий в базе данных:")
    categories = {}
    for model in all_models[:100]:  # Первые 100 для примера
        text = (model.get('title', '') + ' ' + model.get('description', '')).lower()
        category = analyzer.classify_image_by_text(text)
        categories[category] = categories.get(category, 0) + 1
    
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"  {analyzer._get_category_icon(cat)} {cat}: {count}")
    
    print("\n🔍 Тестирование категориального определения:")
    
    # Тест 1: Стул
    test_image = create_test_image()
    
    # Имитируем результат определения категории
    print("\n1️⃣ Тест поиска стула:")
    print("   🎯 Категория запроса: furniture")
    
    # Демонстрация улучшенного поиска
    embedding = analyzer.process_image(test_image)
    if embedding is not None:
        results = analyzer.find_similar_models_enhanced(embedding, 'furniture', 5)
        
        print(f"   ✅ Найдено результатов: {len(results)}")
        for i, result in enumerate(results[:3], 1):
            similarity = result.get('similarity_score', 0)
            visual = result.get('visual_score', 0)
            bonus = result.get('category_bonus', 0)
            category = result.get('detected_category', 'unknown')
            
            print(f"   {i}. {analyzer._get_category_icon(category)} {result['title'][:40]}...")
            print(f"      📊 Общий скор: {similarity:.3f}")
            print(f"      👁️ Визуальный: {visual:.3f}")
            print(f"      🎯 Категор. бонус: {bonus:.3f}")
            print(f"      🏷️ Категория: {category}")
            print()
    
    print("\n2️⃣ Тест обычного поиска (без категории):")
    print("   🎯 Категория запроса: unknown")
    
    if embedding is not None:
        results_old = analyzer.find_similar_models(embedding, 5)
        
        print(f"   ✅ Найдено результатов: {len(results_old)}")
        for i, result in enumerate(results_old[:3], 1):
            similarity = result.get('similarity_score', 0)
            category = result.get('detected_category', 'unknown')
            
            print(f"   {i}. {analyzer._get_category_icon(category)} {result['title'][:40]}...")
            print(f"      📊 Скор: {similarity:.3f}")
            print(f"      🏷️ Категория: {category}")
            print()
    
    print("\n💡 Ключевые улучшения:")
    print("   • Автоматическое определение категории запроса через CLIP")
    print("   • Бонус +15% за точное совпадение категории")
    print("   • Бонус +8% за родственные категории") 
    print("   • Визуальный + категориальный скор")
    print("   • Более релевантные результаты для мебели и интерьера")
    
    print("\n🎯 Результат: улучшенный алгоритм должен показывать стулья")
    print("   вместо потолков и зеркал для фото стула!")

def test_category_classification():
    """Тестирование классификации по тексту"""
    print("\n🧪 Тестирование классификации по тексту:")
    
    analyzer = ImageAnalyzer()
    
    test_texts = [
        "77168376834851bca1a5 Decorative ceiling Wave ceiling",
        "Современный стул из дерева для кухни",
        "2071021.5b5ee8b6d2811 Promemoria - Michelle mirror",
        "Кресло офисное черное кожаное",
        "9482552.58c7c83fab94a B&BMAXALTO3",
        "Лампа настольная LED современная",
        "Шкаф-купе белый глянец 3 двери"
    ]
    
    for text in test_texts:
        category = analyzer.classify_image_by_text(text)
        icon = analyzer._get_category_icon(category)
        print(f"   {icon} \"{text[:50]}...\" → {category}")

if __name__ == "__main__":
    # Запуск тестов
    test_category_classification()
    asyncio.run(test_improved_search()) 