
LLM для Zero-Shot рекомендаций и профилирования пользователей

1. Hou, Y., Zhang, J., Lin, Z., Lu, H., Xie, R., McAuley, J., & Zhao, W. X. (2024). Large Language Models are Zero-Shot Rankers for Recommender Systems. ECIR 2024.
Исследование показывает zero-shot способности LLM в ранжировании элементов для рекомендаций. LLM могут эффективно ранжировать кандидатов без обучения на целевом датасете. Это базовая работа для понимания zero-shot возможностей LLM в рекомендациях
https://arxiv.org/abs/2305.08845

2. Bang, S., et al. (2025). LLM-based User Profile Management for Recommender System. arXiv:2502.14541
Предлагает фреймворк PURE для построения и поддержания эволюционирующих профилей пользователей на основе LLM. Включает компоненты Review Extractor, Profile Updater, Recommender. Напрямую связано с генерацией пользовательских профилей через LLM
https://arxiv.org/abs/2502.14541

3. Kim, S., Kang, H., Choi, S., Kim, D., Yang, M., & Park, C. (2024). Large language models meet collaborative filtering: An efficient all-round llm-based recommender system. RecLM
Создание профилей пользователей и элементов с использованием LLM для cold-start сценариев. Использует instruction-tuning LLM для захвата коллаборативной информации. Генерация zero-shot профилей с сильной обобщающей способностью
https://arxiv.org/html/2412.19302v3

4. ZeroPOIRec: Large language models are zero-shot point-of-interest recommenders (2025). Data Mining and Knowledge Discovery
Zero-shot POI рекомендации с использованием пространственно-временных паттернов. Profiler module для извлечения многоаспектных профилей пользователей из истории посещений. Подход к zero-shot профилированию без истории взаимодействий
https://link.springer.com/article/10.1007/s10618-025-01148-w


Когнитивное моделирование и психология в рекомендательных системах

5. Lex, E., & Schedl, M. (2022). Psychology-informed Recommender Systems. Foundations and Trends in Information Retrieval
Comprehensive review систем, использующих когнитивные процессы, личность и аффективные сигналы. Рассматривает cognition-inspired, personality-aware, affect-aware recommender systems. Теоретическая база для когнитивного моделирования в рекомендациях
https://elisabethlex.info/docs/2021fntir-psychology.pdf

6. Kunkel, J., Ngo, T., Ziegler, J., & Krämer, N. (2021). Identifying Group-Specific Mental Models of Recommender Systems: A Novel Quantitative Approach. INTERACT 2021
Количественный подход к идентификации ментальных моделей пользователей рекомендательных систем. Использует card sorting для выявления процедурных vs концептуальных моделей. Помогает понять когнитивные модели пользователей при взаимодействии с системами
https://link.springer.com/chapter/10.1007/978-3-030-85610-6_23

7. Andjelkovic, I., et al. (2019). Mood-aware recommender systems: Survey of affective recommender systems
Обзор аффективных рекомендательных систем, учитывающих настроение и эмоции. Рассматривает балансировку когнитивной нагрузки, удовлетворенность, качество рекомендаций. Моделирование эмоционального состояния для адаптации рекомендаций
https://arxiv.org/html/2508.20289v1

8. Van der Veen, J., et al. (2022). Three levels at which the user's cognition can be represented in artificial intelligence. Frontiers in Artificial Intelligence
Три уровня интеграции когнитивных моделей в AI: implementation, algorithmic, computational. Фреймворк для представления когнитивных состояний пользователей
https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2022.1092053/full

9. Jawaheer, G., Weller, P., & Kostkova, P. (2014). Modeling User Preferences in Recommender Systems. ACM TIIS
Классификационный фреймворк для explicit/implicit feedback с учетом когнитивных усилий. Рассматривает свойства Cognitive Effort, User Model, Scale of Measurement, Domain Relevance. Учет когнитивной нагрузки при сборе пользовательских предпочтений
https://www.researchgate.net/publication/263766194


Контекстно-зависимые и сессионные рекомендательные системы

10. Wang, S., Cao, L., et al. (2021). A Survey on Session-based Recommender Systems. ACM Computing Surveys
Систематический обзор session-based рекомендательных систем. Рассматривает краткосрочные динамические предпочтения, контекстуальная зависимость. Захват текущих намерений пользователя в реальном времени
https://arxiv.org/pdf/1902.04864

11. Latifi, S., Mauro, N., & Jannach, D. (2021). Session-aware recommendation: A surprising quest for the state-of-the-art. Information Sciences
Benchmark session-aware алгоритмов. Показывает, что простые методы на основе nearest neighbors превосходят нейронные подходы. Критический анализ методологии оценки session-based систем
https://www.sciencedirect.com/science/article/pii/S0020025521005089

12. Zhang, L., et al. (2022). Context-aware session recommendation based on recurrent neural networks. ScienceDirect
SCAR модель, интегрирующая контекстуальную информацию (геолокация, время) в RNN. Комбинация Add, Stack, MLP для слияния низкоразмерных векторных признаков. Использование контекстуальных сигналов для адаптации рекомендаций
https://www.sciencedirect.com/science/article/abs/pii/S0045790622001987

13. Kulkarni, S., & Rodd, S. F. (2020). Context Aware Recommendation Systems: A review of the state of the art techniques. Computer Science Review
Обзор context-aware систем с классификацией контекста и стратегий включения. Категории контекста: Temporal, spatial, social, device-based. Фреймворк для многомерного контекстуального моделирования
https://www.sciencedirect.com/science/article/abs/pii/S1574013719301406

14. Jannach, D., Mobasher, B., & Berkovsky, S. (2020). Research directions in session-based and sequential recommendation. User Modeling and User-Adapted Interaction
Направления исследований в session-aware рекомендациях. Гибридные системы, использование side information, долгосрочные vs краткосрочные предпочтения. Интеграция контекстуальной информации для cold-start сценариев
https://link.springer.com/article/10.1007/s11257-020-09274-4


Мультимодальное и адаптивное обучение

15. Li, Y., & Lu, B. (2025). Intelligent educational systems based on adaptive learning algorithms and multimodal behavior modeling. PeerJ Computer Science
Мультимодальный fusion (MMF) алгоритм для интеграции гетерогенных данных о поведении. Архитектура: Stacked denoising autoencoders + Restricted Boltzmann Machines. Реал-тайм адаптация на основе мультимодальных характеристик пользователя
https://peerj.com/articles/cs-3157/

16. Essa, S. G., Çelik, T., & Human-Hendricks, N. E. (2025). Artificial intelligence in adaptive education: a systematic review. Discover Education
Систематический обзор AI-powered adaptive learning с emphasis на ML, DL, multimodal analytics. Рассматривает XAI, modular designs, culturally responsive frameworks. Адаптация контента к индивидуальным профилям в реальном времени
https://link.springer.com/article/10.1007/s44217-025-00908-6

17. CDARS (2025). Enhancing Recommendation Systems with Real-Time Adaptive Learning and Multi-Domain Knowledge Graphs. MDPI
Cross-domain adaptive recommendation system с real-time behavioral tracking. Инновации: Explainable Adaptive Learning (EAL) module, multimodal sentiment analysis. Динамическая адаптация на основе контекстуальных факторов
https://www.mdpi.com/2504-2289/9/5/124

18. Chen, L., et al. (2025). Adaptive deep reinforcement learning for personalized learning pathways. ScienceDirect
A-DRL платформа для интеллектуальных рекомендаций персонализированных траекторий обучения. Multimodal fusion (performance, emotion, cognition) + real-time feedback. Динамическая адаптация траекторий на основе обратной связи
https://www.sciencedirect.com/science/article/pii/S2666920X25001031

19. Zhao, Z. (2025). AI-Driven Learning-Style Detection for Personalized MOOC Content Delivery. International Journal of Education and Social Development
AI-алгоритмы для автоматической идентификации learning style preferences. Точность часто выше 90% в определении стилей обучения. Автоматическое определение стилей восприятия информации
https://ijesd.com/index.php/ijesd/article/view/141


Few-Shot и Meta-Learning для Cold-Start

20. Hao, B., et al. (2020). Few-Shot Representation Learning for Cold-Start Users and Items. APWeb-WAIM 2020
Формулировка cold-start как few-shot learning задачи. Attention-based encoder для агрегации K training instances. Предсказание представлений cold-start пользователей на основе ограниченных данных
https://xiaojingzi.github.io/publications/APWeb20-HAO-et-al-FewshotRec.pdf

21. Li, M., Hu, S., Zhu, F., & Zhu, Q. (2024). Few-Shot Learning for Cold-Start Recommendation. LREC-COLING 2024
Иерархическая структура: global-meta, local-meta, specific parameters. Учет локальных зависимостей между пользователями (не глобальная независимость). Адаптивная кластеризация локальных пользователей
https://aclanthology.org/2024.lrec-main.631/

22. Zhang, Y., et al. (2024). Content-Aware Few-Shot Meta-Learning for Cold-Start Recommendation. Sensors
CFSM модель с double-tower network (DT-Net). Meta-encoder для user representations + mutual attention encoder для items. Model-agnostic meta-optimization для быстрой онлайн адаптации
https://www.mdpi.com/1424-8220/24/17/5510

23. Yang, H., et al. (2025). Instructional Prompt Optimization for Few-Shot LLM-Based Recommendation. arXiv:2509.09066
Оптимизация промптов для few-shot LLM-рекомендаций в cold-start сценариях. Структура промпта: Instructional header + support set (2-10 similar profiles) + cold-start user metadata. Результаты: +18.7% Precision@5, +21.3% NDCG@10 vs zero-shot LLM на Amazon. Эффективная адаптация без fine-tuning через prompt engineering
https://arxiv.org/pdf/2509.09066


Базовые алгоритмы для сравнения

24. Content-based filtering
Подход на основе атрибутов элементов. Ограничения в cold-start сценариях из-за sparse metadata.

25. Popularity-based
Рекомендация популярных элементов новым пользователям. Baseline для cold-start оценки.

26. Rendle, S., et al. (2009). BPR: Bayesian Personalized Ranking from Implicit Feedback. UAI 2009
Байесовская персонализированная матричная факторизация

27. Kang, W.-C., & McAuley, J. (2018). Self-attentive sequential recommendation. ICDM 2018
SASRec - Self-attention механизм для последовательных рекомендаций
https://arxiv.org/abs/1808.09781



Датасеты

30. MovieLens
MovieLens-1M: 6,040 users, 3,706 movies, около 1M ratings. MovieLens-20M: 138,493 users, 27,278 movies, около 20M ratings. Включает: ratings, timestamps, user demographics, movie metadata
https://grouplens.org/datasets/movielens/

31. Amazon Reviews
Несколько категорий продуктов. Включает: ratings, reviews, product metadata, user information. Подходит для cross-domain и cold-start экспериментов
https://nijianmo.github.io/amazon/index.html


32. serendipity-sac2018

33. Taobao-Serendipity-Dataset-master



Дополнительные работы

44. Wei, W., et al. (2024). LLMRec: Large Language Models with Graph Augmentation for Recommendation. WSDM 2024
Graph augmentation с использованием LLM

45. Zhang, Y., et al. (2020). Explainable Recommendation: A Survey and New Perspectives
Связь объяснений с cognitive science и human decision making