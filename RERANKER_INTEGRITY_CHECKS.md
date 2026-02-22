# Reranker Ranking Integrity Checks

## ✅ Проверки реализованы

### 1. Направление сортировки
- **Проверено**: `src/rerank_llm.py` и `src/rerank_crossencoder.py` используют `reverse=True` (descending)
- **Статус**: ✅ Корректно
- **Доказательство**: Relevant items имеют более высокий score (-4.34 > -4.44)

### 2. Assert'ы после rerank
- **Добавлено в `src/run_experiment.py`**:
  - Assert 1: Проверка сортировки сразу после `reranker.rerank()`
  - Assert 2: Проверка ID-score соответствия
  - Assert 3: Проверка сортировки после `[:topk]` обрезки
  - Assert 4: Проверка финальных результатов перед сохранением
  - Assert 5: Проверка на дубликаты ID (дедупликация)

### 3. Стабильность сортировки (tie-breaks)
- **Добавлено в `src/rerank_llm.py`**:
  - При одинаковых scores используется исходный порядок кандидатов как tie-break
  - Это обеспечивает воспроизводимость результатов

### 4. Дедупликация
- **Проверено**: Дедупликация происходит **до** rerank (в `candidate_retrieval.py`)
- **Статус**: ✅ Корректно - порядок не нарушается после rerank

### 5. ID-Score соответствие
- **Проверено**: Assert'ы проверяют, что ID и score остаются связанными на всех этапах
- **Статус**: ✅ Корректно

### 6. Смешивание источников кандидатов
- **Проверено**: В hybrid mode (BM25+ANN) кандидаты объединяются **до** rerank
- **Статус**: ✅ Корректно - reranker получает единый список и сортирует его

### 7. Комплексный тест
- **Создан**: `tools/comprehensive_reranker_test.py`
- **Проверяет**:
  - Сортировку финальных результатов
  - ID-score соответствие
  - Дедупликацию
  - A/B тест (descending vs ascending)

## 🔍 Места проверки в коде

### `src/rerank_llm.py` (строки 57-63)
```python
scores = self.model.predict(pairs, show_progress_bar=False, batch_size=batch_size)
scored = list(zip(ids, scores))
# Sort in DESCENDING order with stable tie-break
scored_with_original_rank = [(id, score, orig_idx) for orig_idx, (id, score) in enumerate(scored)]
scored_with_original_rank.sort(key=lambda x: (x[1], -x[2]), reverse=True)
scored = [(id, score) for id, score, _ in scored_with_original_rank]
```

### `src/run_experiment.py` (строки 246-283)
- Assert после rerank (строка ~250)
- Assert ID-score соответствие (строка ~255)
- Assert после topk (строка ~256)
- Assert финальных результатов (строка ~280)
- Assert дедупликации (строка ~285)

## 🧪 Запуск тестов

### Комплексный тест целостности
```bash
python tools/comprehensive_reranker_test.py
```

### Тест стабильности
```bash
python tools/test_reranker_stability.py
```

### Проверка сортировки
```bash
python tools/check_reranker_sorting.py <run_id>
```

## ⚠️ Потенциальные проблемы (проверены)

1. ✅ **Обрезка до сортировки**: Проверено - `[:topk]` применяется **после** сортировки
2. ✅ **Потеря порядка через set/dict**: Проверено - set используется только для дедупликации **до** rerank
3. ✅ **Несовпадение id ↔ score**: Проверено - assert'ы проверяют соответствие
4. ✅ **Смешивание источников**: Проверено - гибридный retrieval объединяет **до** rerank
5. ✅ **Tie-breaks**: Исправлено - добавлен стабильный tie-break по исходному порядку
6. ✅ **Смена знака score**: Проверено - нет инверсии или нормализации после rerank

## 📊 Результаты проверок

Все проверки пройдены:
- ✅ Сортировка корректна на всех этапах
- ✅ ID-score соответствие сохранено
- ✅ Нет дубликатов после rerank
- ✅ Стабильность обеспечена через tie-break

## 🔧 Рекомендации

1. **Мониторинг**: Assert'ы будут логировать предупреждения при обнаружении проблем
2. **Тестирование**: Регулярно запускать `comprehensive_reranker_test.py` перед релизами
3. **Воспроизводимость**: Использовать стабильный tie-break для одинаковых scores
