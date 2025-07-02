from __future__ import annotations

"""Factory module, simplified.
If Firebase modules присутствуют – можно расширить, но пока используем SQLite.
"""

from typing import Dict
from config import DATABASE_TYPE

# Локальная SQLite реализация
from database import ModelsDatabase


def get_database() -> ModelsDatabase:
    """Возвращает экземпляр активной базы данных (сейчас всегда SQLite)."""
    # В будущем можно переключать на Firebase по DATABASE_TYPE
    return ModelsDatabase()


def get_database_info() -> Dict[str, str | int]:
    """Краткая информация о текущей базе данных."""
    db = get_database()
    return {
        "type": DATABASE_TYPE,
        "models_count": db.get_models_count(),
    } 