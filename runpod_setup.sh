#!/bin/bash

echo "🚀 Настройка 3D Model Bot на RunPod"

# Переходим в папку проекта
cd /workspace/3D-model-bot

# Обновляем репозиторий
echo "📥 Обновляем код с GitHub..."
git pull origin main

# Проверяем файлы
echo "📂 Проверяем файлы..."
ls -la | grep -E "(bot_lite|image_analyzer_lite|config)"

# Устанавливаем переменные окружения для экономии памяти
echo "⚙️ Настраиваем окружение..."
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Проверяем пакеты
echo "📦 Проверяем пакеты..."
python -c "import torch; print(f'✅ PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'✅ Transformers: {transformers.__version__}')"
python -c "import sklearn; print('✅ Scikit-learn')"

# Тестируем конфиг
echo "🔧 Проверяем конфигурацию..."
python -c "import config; print('✅ Config загружен')"

# Запускаем бота
echo "🤖 Запускаем бота..."
python bot_lite.py 