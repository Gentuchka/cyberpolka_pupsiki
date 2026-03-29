# 🏦 Data Fusion Contest 2026 — Задача 2 «Киберполка»

### Команда: **cyberpolka_pupsiki** 
**Авторы** [Gentuchka](github.com/Gentuchka), [Svetankova](https://github.com/Svetankova)




**Макро-задача:** Multi-label классификация — предсказание вероятности владения 41 финансовым продуктом банка на основе обезличенных данных клиентов.

**Финальный результат:** `Macro OOF AUC = 0.8328` (rank blend на валидации)

**Статус:** ✅ Решение завершено, пайплайн воспроизводим

**Вдохновлялись**  https://github.com/d-ulybin/data_fusion_track_2 (null_pattern_pca), https://github.com/artyerokhin/datafusion2026_public_task2 (идея стекинга)

---

## 📋 Содержание

1. [Вводные данные и ограничения](#1-вводные-данные-и-ограничения)
2. [Архитектура решения](#2-архитектура-решения)
3. [Параметры моделей](#3-параметры-моделей)
4. [Особенности и ключевые решения](#4-особенности-и-ключевые-решения)
5. [Структура репозитория](#5-структура-репозитория)
6. [Инструкция по запуску](#6-инструкция-по-запуску)
7. [Технические требования](#7-технические-требования)

---

## 1. Вводные данные и ограничения

| Параметр | Значение |
| :--- | :--- |
| **Обучающая выборка** | ~1,000,000 клиентов |
| **Тестовая выборка** | ~250,000 клиентов |
| **Целевые переменные** | 41 бинарный таргет (multi-label) |
| **Extra-признаки** | ~1000+ (отобрано 700 → 300) |
| **Семантика признаков** | ❌ Не известна (анонимизированные данные) |
| **Платформа** | Kaggle (GPU T4/P100, 16 ГБ RAM) |
| **Метрика** | Macro ROC-AUC по 41 таргету |

**Ключевые ограничения:**
- **16 ГБ RAM** на Kaggle — требует бережной работы с памятью (Polars lazy, type downcasting Float64→Float32, Int64→Int32)
- Отсутствие бизнес-контекста исключает осмысленный domain-specific Feature Engineering
- Обучение 41 × 5 folds × 2 моделей × 5 seeds = **2050 моделей** — необходима GPU-акселерация

---

## 2. Архитектура решения

Стекинг с multilabel-стратифицированной 5-fold кросс-валидацией:

```
┌─────────────────────────────────────────────────────────────────┐
│                        ВХОДНЫЕ ДАННЫЕ                           │
│  train_main + train_extra (~1000+ признаков) + 41 targets       │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              NB00: СХЕМА И ФОЛДЫ                                │
│  MultilabelStratifiedKFold (n_splits=5, seed=42)                │
│  Отклонение частоты таргетов по фолдам < 2%                     │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              NB01: ОТБОР ПРИЗНАКОВ (LightGBM Gain)              │
│  Быстрый скан на 200K сэмпле → top700 (L1) / top300 (L2)      │
│  LightGBM: lr=0.10, leaves=31, 50 rounds, GPU                  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              NB02: БАЗОВЫЕ ПРИЗНАКИ (~750 фичей)                │
│  Categorical: code (Int32) + freq (UInt32)                      │
│  Null features: count & ratio per block                         │
│  Row statistics: mean, std, min, max, nonzero per block         │
│  Extra top-700: все отобранные extra-признаки                   │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                  ┌────────────┴────────────┐
                  ▼                         ▼
┌────────────────────────────┐  ┌────────────────────────────────┐
│  NB03: L1 OOF LightGBM    │  │  NB04: GLOBAL AGGS + NULL SVD  │
│  НАМЕРЕННО СЛАБАЯ модель   │  │  7 глобальных агрегаций        │
│  num_leaves=15, depth=4    │  │  20 SVD-компонент из           │
│  75 rounds, БЕЗ early stop │  │  null-паттернов (train+test)   │
│  → 41 мета-признак (OOF)   │  │  → 27 доп. фичей              │
└────────────┬───────────────┘  └───────────────┬────────────────┘
             │                                  │
             └──────────────┬───────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│         L2: ФИНАЛЬНЫЕ МОДЕЛИ (~817 фичей на таргет)             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Для каждого из 41 таргетов:                            │    │
│  │  Features = Base(750) + Aggs(7) + SVD(20) + Meta(40*)   │    │
│  │  * исключая мета-признак своего таргета (anti-leakage)  │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │  NB05: L2 LightGBM — lr=0.03, leaves=127, 3000 rounds  │    │
│  │  NB06: L2 CatBoost — lr=0.03, depth=7, 3000 iterations │    │
│  │  Оба: 5-seed averaging (42, 777, 555, 2026, 404)       │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              NB07: RANK BLEND                                   │
│  LGBM (65%) + CatBoost (35%) через ранговое усреднение          │
│  Macro OOF AUC = 0.8386                                        │
│  → submission_RANK_BLEND.parquet                                │
└─────────────────────────────────────────────────────────────────┘
```

**Ключевая идея стекинга:** слабая L1-модель (num_leaves=15, без early stopping) генерирует разнообразные, некоррелированные мета-признаки. Сильная L2-модель учится комбинировать эти слабые сигналы вместе с базовыми фичами, достигая Macro OOF AUC = 0.8328 на отдельном CatBoost, и **0.8386** после rank blend с LightGBM.

---

## 3. Параметры моделей

### L1 LightGBM (Weak Meta-Learner)

| Параметр | Значение | Комментарий |
| :--- | :--- | :--- |
| `objective` | `binary` | Бинарная классификация per-target |
| `learning_rate` | 0.10 | Агрессивный LR для быстрого обучения |
| `num_leaves` | **15** | **Намеренно слабый** (vs стандартных 31-127) |
| `max_depth` | **4** | Ограничение глубины для underfitting |
| `num_boost_round` | **75** | Фиксированное число раундов |
| `early_stopping` | **Нет** | Принципиально БЕЗ early stopping |
| `feature_fraction` | 0.80 | Рандомизация для разнообразия |
| `bagging_fraction` | 0.80 | Рандомизация для разнообразия |
| `min_data_in_leaf` | 100 | Регуляризация |
| `lambda_l2` | 1.0 | L2-регуляризация |
| `device` | GPU (CUDA) | |

> **Зачем слабая L1?** Сильная L1 запоминает train → мета-признаки становятся «копией» таргета → L2 переобучается. Слабая L1 даёт разнообразные, некоррелированные сигналы — идеальные входы для мощной L2.

### L2 LightGBM (Strong Final Model)

| Параметр | Значение |
| :--- | :--- |
| `objective` | `binary` |
| `learning_rate` | 0.03 |
| `num_leaves` | 127 |
| `feature_fraction` | 0.70 |
| `bagging_fraction` | 0.80 |
| `min_data_in_leaf` | 100 |
| `lambda_l2` | 1.0 |
| `max_bin` | 63 |
| `num_boost_round` | 3000 |
| `early_stopping_rounds` | 150 |
| `device` | GPU |
| **Seed Averaging** | 5 seeds: [42, 777, 555, 2026, 404] |

### L2 CatBoost (Strong Final Model)

| Параметр | Значение |
| :--- | :--- |
| `loss_function` | `Logloss` |
| `eval_metric` | `AUC` |
| `learning_rate` | 0.03 |
| `depth` | 7 |
| `l2_leaf_reg` | 3.0 |
| `random_strength` | 1.0 |
| `bagging_temperature` | 1.0 |
| `border_count` | 254 |
| `iterations` | 3000 |
| `early_stopping_rounds` | 200 |
| `nan_mode` | `Min` |
| `auto_class_weights` | `Balanced` |
| `task_type` | GPU |
| **Seed Averaging** | 5 seeds: [42, 777, 555, 2026, 404] |

### Rank Blend

| Параметр | Значение |
| :--- | :--- |
| **Вес LGBM** | 0.65 |
| **Вес CatBoost** | 0.35 |
| **Метод** | rank-based blend (rankdata → нормализация → взвешенная сумма) |

---

## 4. Особенности и ключевые решения

### ✅ Что сработало

| Метод | Эффект | Комментарий |
| :--- | :--- | :--- |
| **Стекинг** | Core прирост | Слабый L1 + сильный L2 — основной драйвер качества |
| **MultilabelStratifiedKFold** | Стабильность CV | Балансирует все 41 таргет одновременно |
| **Rank Blend (LGBM + CatBoost)** | +0.005-0.006 | 0.8328 → 0.8386 благодаря ранговому блендингу |
| **Anti-leakage в мета-фичах** | Корректность | Для таргета T исключаем meta_T (40 мета-фичей из 41) |
| **Null Pattern SVD** | Дополнительный сигнал | Паттерны пропусков как латентные фичи |
| **5-Seed Averaging** | Стабильность test | Усреднение по 5 сидам на тестовой выборке |
| **Category code + freq encoding** | Для CatBoost | Code как нативные cat_features + частотное кодирование |

### 💡 Инсайты

1. **Модель L1 должна быть максимально простой и непереобученной** — это принципиально для стекинга. Сильная L1 убивает разнообразие мета-признаков.
2. **Идея стекинга работает** — даже при OOF AUC ~0.70-0.75 на L1, итоговый L2 достигает 0.83+
3. **Rank blend устойчивее арифметического** — нивелирует разницу в калибровке моделей
4. **Null-паттерны информативны** — SVD на матрице пропусков даёт дополнительные 20 фичей

---

## 5. Структура репозитория

### Скрипты (основной пайплайн)

```
cyberpolka_pupsiki/
├── 00_schema_and_folds.py          # Схема данных, MultilabelStratifiedKFold (5 фолдов)
├── 01_feature_selection.py         # Отбор фичей: top700 / top300 по LGBM gain
├── 02_build_base_features.py       # Сборка базовых фичей (~750): encoding, nulls, stats
├── 03_build_meta_oof_lgbm.py       # L1 OOF: слабый LightGBM → 41 мета-признак
├── 04_build_global_aggs_null_svd.py  # Глобальные агрегации (7) + Null SVD (20)
├── 05_train_final_lgbm_meta.py     # L2 LightGBM: сильная модель + 5-seed averaging
├── 06_train_final_catboost_meta.py # L2 CatBoost: сильная модель, OOF + test
├── 07_rank_blend.py                # Rank blend LGBM(65%) + CatBoost(35%) → submission
├── README.md                       # Этот файл
└── prepared/                       # Артефакты (создаётся автоматически)
    └── artifacts/
        ├── config/                 # folds.parquet, target_cols.json, schema.json
        ├── features/               # train/test_base_features.parquet, base_feature_cols.json
        ├── meta/                   # l1_lgbm_oof.parquet, l1_lgbm_test.parquet
        ├── aggs/                   # global_aggs.parquet, null_svd.parquet
        ├── preds/                  # final_lgbm_*.parquet, final_catboost_*.parquet
        ├── logs/                   # scores, summaries
        └── submission/             # submission_RANK_BLEND.parquet
```

### Исходные ноутбуки (Kaggle)

```
cyberpolka_pupsiki/
├── 00-schema-and-folds2803.ipynb
├── 01-feature-selection-41targets2803.ipynb
├── 02-build-base-features2803.ipynb
├── 03-build-meta-oof-lgbm-memory-safe2803.ipynb
├── 04-build-global-aggs-and-null-features2803.ipynb
└── 06-train-final-catboost-meta.ipynb

vika/
├── 00-schema-and-folds.ipynb
├── 01-feature-selection-41targets-randomsample.ipynb
├── 02-build-base-features.ipynb
├── 03-build-meta-oof-5fold-cuda-safe.ipynb
├── 03-build-meta-oof-lgbm-memory-safe.ipynb
├── 04-build-global-aggs-and-null-features-train-test.ipynb
├── 05-train-final-lgbm-meta.ipynb
├── 06-train-final-catboost-meta.ipynb
├── 06-nn-embeddings-multi-task.ipynb
└── 07-rank-blend.ipynb
```

---

## 6. Инструкция по запуску

Решение можно запустить как набор `.py`-скриптов последовательно. Данные (parquet-файлы) должны лежать в текущей директории.

### Порядок запуска

```bash
python 00_schema_and_folds.py          # → folds.parquet, target_cols.json, schema.json
python 01_feature_selection.py         # → top700/top300 feature lists
python 02_build_base_features.py       # → train_base_features.parquet, test_base_features.parquet
python 03_build_meta_oof_lgbm.py       # → l1_lgbm_oof.parquet, l1_lgbm_test.parquet
python 04_build_global_aggs_null_svd.py  # → train/test_global_aggs.parquet, null_svd features
python 05_train_final_lgbm_meta.py     # → final_lgbm_oof.parquet, final_lgbm_test.parquet
python 06_train_final_catboost_meta.py # → final_catboost_oof.parquet, final_catboost_test.parquet
python 07_rank_blend.py                # → submission_RANK_BLEND.parquet (Macro OOF AUC = 0.8386)
```

Все промежуточные артефакты сохраняются в `prepared/artifacts/`.

### Данные

Исходные данные загружаются с [платформы соревнования](https://ods.ai/tracks/data-fusion-2026-competitions/competitions/data-fusion2026-cybershelf):
- `train_main.parquet` — основные признаки
- `train_extra.parquet` — дополнительные признаки
- `train_target.parquet` — 41 бинарный таргет
- `test_main.parquet`, `test_extra.parquet` — тестовая выборка
- `sample_submit.parquet` — формат сабмита

---

## 7. Технические требования

| Компонент | Минимум (Kaggle) |
| :--- | :--- |
| **RAM** | 16 ГБ |
| **GPU** | NVIDIA T4 / P100 (CUDA) |
| **Python** | 3.10+ |

### Зависимости

```
catboost==1.2.9
lightgbm==4.6.0
scikit-learn==1.8.0
polars==1.38.1
pandas==3.0.0
numpy==2.4.2
pyarrow==23.0.1
scipy==1.17.0
category_encoders==2.9.0
iterstrat  # MultilabelStratifiedKFold
```

---

> **Ссылка на соревнование:** [Data Fusion 2026 — Киберполка](https://ods.ai/tracks/data-fusion-2026-competitions/competitions/data-fusion2026-cybershelf)
