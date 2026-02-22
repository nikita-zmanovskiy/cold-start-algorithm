# Finalization Checklist for Diagnostic Paper

## ✅ Обязательные задачи (Must-Have)

### 1. Исправить GT / исключить проблемных пользователей
- [x] Создан `tools/fix_gt_catalog.py`
- [ ] Запустить: `python -m tools.fix_gt_catalog`
- [ ] Заменить `ground_truth.json` на `ground_truth_fixed.json`
- [ ] Пересобрать эксперименты с исправленным GT

### 2. Сохранить reranker scores
- [x] Модифицирован `src/run_experiment.py` для сохранения всех scores
- [x] Scores сохраняются в `results/*.json` как `reranker_scores`
- [ ] Пересобрать эксперименты для получения scores

### 3. Recall@K curves и таблицы
- [x] Уже реализовано в `tools/hypothesis_analysis.py`
- [x] Генерирует `coverage_recall_curves.png`
- [ ] Включить в статью как Figure

### 4. Exposure metrics
- [x] Уже реализовано в `tools/hypothesis_analysis.py`
- [x] Генерирует `top1_exposure_*.png`, вычисляет Gini, entropy
- [ ] Включить в Results/Diagnostics

### 5. Ablation: rerank на разных pool sizes
- [x] Создан `src/run_ablation_pool_sizes.py`
- [ ] Запустить: `python -m src.run_ablation_pool_sizes --pool-sizes 200 500 1000`
- [ ] Сравнить HR/ndcg для разных pool sizes

### 6. Статистические тесты и CI
- [x] Реализовано в `tools/enhanced_stat_tests.py`
- [x] Вычисляет paired t-test, Wilcoxon, Cohen's d, 95% CI
- [ ] Включить в Results таблицей

### 7. Error Analysis
- [x] Создан `tools/error_analysis.py`
- [ ] Запустить: `python -m tools.error_analysis`
- [ ] Выбрать 5-10 примеров для статьи
- [ ] Написать краткие объяснения для каждого

### 8. Master run file
- [x] Реализовано в `tools/build_master_results.py`
- [x] Создает `experiments/master_results.json`
- [ ] Проверить, что все поля присутствуют
- [ ] Использовать для Appendix

## 📋 Рекомендуемые улучшения

### 9. Больше seeds и стабильность
- [ ] Запустить 5+ seeds для ключевых конфигураций
- [ ] Проверить стабильность результатов

### 10. Score distributions анализ
- [x] Создан `tools/analyze_scores.py`
- [ ] Запустить: `python -m tools.analyze_scores`
- [ ] Построить histograms и scatter plots
- [ ] Вычислить Spearman correlation

### 11. Простые улучшения
- [ ] Попробовать rerank только top-200
- [ ] Попробовать score calibration (z-score, min-max)
- [ ] Попробовать diversity penalty
- [ ] Попробовать hybrid scoring (popularity + CE)

### 12. Дополнительные baselines
- [ ] BM25 retrieval
- [ ] Hybrid retrieval (BM25 + FAISS)
- [ ] Simple learning-to-rank

### 13. Runtime analysis
- [ ] Измерить скорость rerank
- [ ] Измерить использование памяти
- [ ] Включить в Results

## 🚀 Порядок выполнения

1. **Исправить GT:**
   ```bash
   python -m tools.fix_gt_catalog
   # Заменить ground_truth.json
   ```

2. **Пересобрать основные эксперименты:**
   ```bash
   python -m src.run_all_experiments --n-users 500 --seeds 42 7 123
   ```

3. **Запустить ablation по pool sizes:**
   ```bash
   python -m src.run_ablation_pool_sizes --pool-sizes 200 500 1000
   ```

4. **Собрать master results:**
   ```bash
   python -m tools.build_master_results
   ```

5. **Провести анализ:**
   ```bash
   python -m tools.hypothesis_analysis
   python -m tools.error_analysis
   python -m tools.enhanced_stat_tests
   ```

6. **Сгенерировать визуализации:**
   ```bash
   python -m tools.plotting
   python -m tools.advanced_plotting
   ```

7. **Создать таблицы для статьи:**
   ```bash
   python -m tools.generate_paper_tables
   ```

## 📊 Что включить в статью

### Figures:
- Recall@K curves (из hypothesis_analysis)
- Boxplots per-user HR@10/nDCG@10 (из advanced_plotting)
- Top-1 exposure plots (из hypothesis_analysis)
- GT positions histogram (из hypothesis_analysis)
- Recall vs HR scatter (из hypothesis_analysis)

### Tables:
- Main results (HR@10, nDCG@10) с CI (из generate_paper_tables)
- Statistical tests (из enhanced_stat_tests)
- Diagnostics (recall@K, unique_top1, Gini, entropy)
- Ablation results (pool sizes)

### Text:
- Error analysis examples (5-10 cases из error_analysis)
- Limitations section
- Discussion о причинах проблем

### Appendix:
- Master results JSON
- Per-user CSVs
- Configuration files
- Reproduction instructions
