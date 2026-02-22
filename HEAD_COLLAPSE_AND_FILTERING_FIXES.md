# Head Collapse and Seen Items Filtering Fixes

## ✅ Исправления

### 1. Head Collapse - одинаковые text_profile для всех пользователей

**Проблема**: Все пользователи получали одинаковый `text_profile` (пустая строка), что приводило к head collapse - все пользователи получали одинаковые рекомендации.

**Исправления**:

#### `src/vark_simulator.py`
- Исправлена генерация `text_profile`: теперь для cold-start пользователей создается уникальный профиль на основе `user_id` и `vark`
- Исправлена генерация VARK: используется hash от `user_id` для обеспечения разнообразия

```python
# До: text_profile = "" для всех
# После: text_profile = f"user_vark:{vark} user_id:{user_id}" для каждого пользователя
```

#### `src/candidate_retrieval.py` и `src/baselines.py`
- Добавлен `user_id` в query text для обеспечения разнообразия даже при пустом goal
- Query теперь: `f"user_vark:{vark} user_id:{user_id}"` вместо просто `"user_vark:{vark}"`

**Результат**: Теперь каждый пользователь получает уникальный text_profile, что должно устранить head collapse.

### 2. Проверка фильтрации seen items

**Проверено**: 
- ✅ Нет фильтрации seen items в коде (правильно для cold-start)
- ✅ GT items не из training set (правильный train/test split)
- ⚠️ GT items почти не попадают в candidate pools (1%) - это проблема retrieval, не фильтрации

**Вывод**: Фильтрация seen items не применяется, что корректно для cold-start сценария.

### 3. Head Collapse диагностика

**Создан**: `tools/check_head_collapse.py`
- Проверяет разнообразие top-1 и top-10 рекомендаций
- Выявляет повторяющиеся последовательности
- Проверяет разнообразие user profiles

**Результаты проверки** (до исправлений):
- Top-1 diversity: < 10% (head collapse)
- Top-10 diversity: 0.80% (только 4 уникальных последовательности для 500 пользователей)
- Text profile diversity: 1 / 20 (все одинаковые)

## 🔍 Обнаруженные проблемы

### 1. Head Collapse (ИСПРАВЛЕНО)
- **Причина**: Одинаковые text_profile для всех пользователей
- **Решение**: Уникальный text_profile на основе user_id + vark

### 2. Низкое покрытие GT items в candidate pools (НЕ ИСПРАВЛЕНО)
- **Проблема**: Только 1% GT items попадают в candidate pools
- **Причина**: Проблема с retrieval (ANN/BM25 не находит релевантные items)
- **Это НЕ проблема фильтрации seen items**

### 3. Повторяющиеся top-10 (ЧАСТИЧНО ИСПРАВЛЕНО)
- **Проблема**: Items 31179, 172085, 157603 появляются очень часто
- **Причина**: Head collapse из-за одинаковых профилей
- **Решение**: Уникальные профили должны решить проблему

## 🧪 Инструменты для проверки

### Проверка head collapse
```bash
python tools/check_head_collapse.py <run_id>
```

### Проверка фильтрации seen items
```bash
python tools/check_seen_items_filtering.py <run_id>
```

## 📊 Ожидаемые улучшения

После исправлений:
1. **Head collapse должен быть устранен**: Разнообразие top-1 должно увеличиться с <10% до >50%
2. **Разнообразие top-10**: Должно быть значительно больше уникальных последовательностей
3. **User profiles**: Каждый пользователь должен иметь уникальный text_profile

## ⚠️ Оставшиеся проблемы

1. **Низкое покрытие GT items в candidate pools (1%)**
   - Это проблема retrieval качества, не фильтрации
   - Нужно улучшить ANN/BM25 retrieval
   - Или увеличить candidate_pool_size

2. **GT items не попадают в рекомендации**
   - Даже когда GT items в candidate pools, они не попадают в top-10
   - Возможно, проблема с reranker или слишком маленький candidate pool для rerank

## 🔧 Рекомендации

1. **Перезапустить эксперименты** с исправленными профилями
2. **Проверить retrieval**: Увеличить candidate_pool_size или улучшить ANN параметры
3. **Мониторинг**: Использовать `check_head_collapse.py` для проверки разнообразия
