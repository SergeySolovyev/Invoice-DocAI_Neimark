# Invoice DocAI v2 — Итоговый отчёт по проекту

## 1. Цель проекта

Сравнительный анализ подходов к извлечению ключевых полей из сканов чеков (receipts):
- **Vendor** (название компании)
- **Date** (дата)
- **Total** (итоговая сумма)

Сравниваются три пайплайна:
1. **OCR Baseline** — EasyOCR + rule-based extraction
2. **Donut Pretrained** — предобученная модель `philschmid/donut-base-sroie`
3. **Donut Fine-tuned** — модель `naver-clova-ix/donut-base`, дообученная на SROIE

Дополнительно проводится **оценка робастности** — устойчивости моделей к искажениям изображений (имитация пересылки через мессенджер).

---

## 2. Датасет

**SROIE 2019** (Scanned Receipts OCR and Information Extraction):
- **Train**: 626 изображений чеков с разметкой (vendor, date, total)
- **Val/Test**: 347 изображений с разметкой
- Формат: JPG-сканы + TXT-файлы с ground truth

Качество разметки:
| Split | Vendor | Date | Total |
|-------|--------|------|-------|
| Train | 626/626 | 601/626 | 624/626 |
| Val   | 347/347 | 332/347 | 345/347 |

---

## 3. Архитектура проекта

```
invoice_docai/
├── README.md
├── data/sroie/
│   ├── raw/SROIE2019/          # Исходные изображения и разметка
│   │   ├── train/img/, train/entities/
│   │   └── test/img/, test/entities/
│   └── processed/              # Манифесты (CSV)
└── v2/
    ├── RUN_ALL_COLAB.ipynb     # Главный ноутбук — полный пайплайн
    ├── src/docai_utils.py      # Утилиты: нормализация, извлечение, метрики
    ├── outputs/                # Результаты: CSV, PNG-графики
    └── notebooks/              # Отдельные ноутбуки по шагам (01–05)
```

### Ключевые компоненты `docai_utils.py`
- `normalize_date()` — приведение дат к формату YYYY-MM-DD (DD/MM/YY, DD/MM/YYYY, month names)
- `normalize_total()` — приведение сумм к формату X.XX
- `extract_fields_from_lines()` — rule-based извлечение полей из OCR-строк
- `extract_total_from_lines()` — 4-уровневая стратегия поиска суммы (keyword → amount → bottom-half → fallback)
- `messenger_corrupt()` — имитация искажений мессенджера (perspective warp, blur, downscale, JPEG compression)
- `evaluate()` — вычисление precision, recall, F1, exact match по полям
- `build_manifest()` — сборка манифеста изображений с ground truth

---

## 4. Пайплайн (шаги выполнения)

### Шаг 0 — Setup
- Подключение Google Drive (или inline-fallback для `docai_utils.py`)
- Скачивание и распаковка SROIE датасета с Google Drive (434 MB)
- Установка зависимостей: `easyocr`, `transformers`, `sentencepiece`, `accelerate`, `opencv`

### Шаг 1 — Rebuild Manifests
- Построение CSV-манифестов с абсолютными путями к изображениям и ground truth
- Quick mode: train=240, val=80 документов (подвыборка для быстрого прогона)

### Шаг 2 — OCR Baseline
- **EasyOCR** (GPU-accelerated) → получение строк текста
- Rule-based извлечение полей: vendor (первые строки, фильтрация), date (regex), total (keyword-priority + range filter)
- Среднее время инференса: ~1186 мс/документ

### Шаг 3 — Donut Pretrained Inference
- Модель: `philschmid/donut-base-sroie` (809M параметров)
- Многослойный парсер выхода: SROIE-теги → token2json → regex fallback
- Среднее время инференса: ~2453 мс/документ

### Шаг 4 — Donut Fine-tuning
- Базовая модель: `naver-clova-ix/donut-base`
- Добавлены специальные токены: `<s_sroie>`, `<s_company>`, `<s_date>`, `<s_total>` и др.
- **Оптимизации для T4 (16 GB VRAM)**:
  - Уменьшение разрешения 2560×1920 → 1280×960
  - fp16 mixed precision (`torch.amp.autocast` + `GradScaler`)
  - Batch size = 1 с gradient accumulation = 2
  - Освобождение VRAM от pretrained-модели и EasyOCR перед обучением
- 2 эпохи (quick mode), loss: 4.94 → 0.80
- Лучший чекпоинт сохранён автоматически

### Шаг 5 — Robustness Evaluation
- Генерация искажённых изображений (messenger-grade corruption):
  - Perspective warp (8%)
  - Gaussian blur (kernel=5)
  - Downscale (60%)
  - JPEG compression (quality=22)
- Прогон OCR и Donut-FT на 80 искажённых изображениях

### Шаг 6 — Summary & Visualizations
- Сводная таблица метрик по всем пайплайнам
- 3 визуализации: F1 by Field, Degradation, Exact Match

---

## 5. Результаты (Quick Mode, 80 val docs)

### Основная таблица метрик

| Pipeline | Condition | Vendor F1 | Date F1 | Total F1 | Micro F1 |
|----------|-----------|-----------|---------|----------|----------|
| **OCR** | clean | 0.49 | **0.78** | 0.63 | 0.63 |
| **OCR** | corrupted | 0.40 | 0.56 | 0.45 | 0.47 |
| **Donut-PT** | clean | 0.00 | 0.05 | 0.00 | 0.02 |
| **Donut-FT** | clean | **0.82** | 0.63 | **0.78** | 0.75 |
| **Donut-FT** | corrupted | 0.69 | 0.54 | 0.64 | 0.63 |

### Лучший пайплайн по каждому полю (clean)
- **Vendor**: Donut-FT (F1 = 0.82)
- **Date**: OCR Baseline (F1 = 0.78)
- **Total**: Donut-FT (F1 = 0.78)

### Exact Match (финансово-критичные поля)
| Pipeline | Condition | Date EM | Total EM |
|----------|-----------|---------|----------|
| OCR | clean | 64.6% | 46.2% |
| OCR | corrupted | 39.2% | 28.7% |
| Donut-PT | clean | 2.5% | 0.0% |
| Donut-FT | clean | 45.6% | 63.7% |
| Donut-FT | corrupted | 36.7% | 47.5% |

### Робастность (средняя деградация F1)
| Pipeline | Avg F1 Drop |
|----------|-------------|
| OCR | +0.166 (хуже) |
| **Donut-FT** | **+0.119** (устойчивее) |

---

## 6. Обзор литературы

### 6.1. Соревнование SROIE 2019

**ICDAR 2019 Competition on Scanned Receipt OCR and Information Extraction** (Huang et al., 2019) — ключевой бенчмарк для задачи извлечения информации из чеков. Три подзадачи: локализация текста, распознавание текста и извлечение ключевых полей (company, date, address, total). Метрика — Entity-level F1 (exact string match).

### 6.2. Лидерборд SROIE (опубликованные результаты)

| Модель | F1 (%) | Тип | Источник |
|--------|:------:|-----|----------|
| BERT + SPADE | 93.67 | OCR-based | Hong et al., 2022 |
| Donut Fine-tuned | 94.40 | End-to-end | Kim et al., 2022 |
| LayoutLM BASE | 95.11 | OCR-based | Xu et al., 2020 |
| PICK (GCN) | 96.10 | OCR-based | Yu et al., 2021 |
| LayoutLMv2 LARGE | 96.39 | OCR-based | Xu et al., 2021 |
| BROS LARGE | **96.62** | OCR-based | Hong et al., 2022 |
| **Наш OCR Baseline** | **64.59** | OCR-based | This work (quick mode) |
| **Наш Donut-FT** | **74.87** | End-to-end | This work (quick mode) |

### 6.3. Ключевые подходы

**LayoutLM** (Xu et al., 2020, KDD): первая модель, совмещающая текстовые и пространственные (2D layout) эмбеддинги для документов. Добавление координат bounding box к BERT даёт +1.4 F1 на SROIE.

**LayoutLMv2** (Xu et al., 2021, ACL): мультимодальная модель (текст + layout + изображение) с spatial-aware self-attention. Предобучение: masked visual-language modeling.

**BROS** (Hong et al., 2022, AAAI): фокус на относительных позициях текстовых блоков вместо абсолютных координат. Устойчив к перестановке порядка чтения.

**Donut** (Kim et al., 2022, ECCV): OCR-free подход. Swin Transformer энкодер + BART декодер. Не требует внешнего OCR — «читает» пиксели напрямую. Лицензия MIT (коммерческое использование).

**PICK** (Yu et al., 2021, ICPR): граф-based подход. Текстовые сегменты = узлы графа, пространственные связи = рёбра. Graph Convolutional Network для классификации.

### 6.4. Ключевые выводы из литературы

1. **Layout-информация добавляет 2-3 F1 пункта** (BERT 93.7 → LayoutLM 95.1 → BROS 96.6)
2. **SROIE фаворитизирует OCR-based модели** из-за метрики exact string match
3. **Fine-tuning — главный рычаг для Donut**: ~84% (base) → ~94% (fine-tuned)
4. **End-to-end модели торгуют точность на робастность**: меньше cascading errors

---

## 7. Почему модели ведут себя по-разному

### 7.1. Каскадное распространение ошибок в OCR

OCR-пайплайн состоит из 3 независимых стадий, каждая из которых может ошибиться:

```
P(correct) = P(detect) × P(recognize | detect) × P(extract | recognize)
```

При точности каждой стадии 90%: 0.90³ = **72.9%** — значительная потеря.

Donut — это единый дифференцируемый пайплайн (image → structured output), где ошибки не каскадируются.

### 7.2. Почему Vendor — самое сложное поле

- **Открытый словарь**: названия компаний бесконечно разнообразны, нет шаблона для валидации
- **OCR проблема**: эвристика «первая непропущенная строка» путает адреса с названиями компаний
- **Donut преимущество**: модель учится, что название компании — обычно в шапке чека (крупный шрифт, по центру)
- **Наш результат**: OCR F1=0.49 vs Donut-FT F1=0.82

### 7.3. Почему Date — единственное поле, где побеждает OCR

- **Закрытый словарь**: даты следуют конечному набору форматов (DD/MM/YYYY и т.д.)
- **Regex-паттерны** покрывают ~95% форматов дат в SROIE с идеальной точностью
- **Donut проблема**: декодер галлюцинирует неправильные годы (bias от предобучения, например 2018 вместо 2019)
- **OCR**: Precision≈1.0, Recall≈0.65 vs Donut-FT: Precision≈0.47, Recall≈0.95
- **Наш результат**: OCR F1=0.78 vs Donut-FT F1=0.63

### 7.4. Почему Donut выигрывает по Total

- **Семантический контекст**: нужно отличить TOTAL от SUBTOTAL, TAX, номеров телефонов
- **OCR проблема**: fallback «наибольшее число» иногда выбирает телефонные номера / ID
- **Donut понимает структуру**: итого обычно внизу документа, после списка товаров
- **Наш результат**: OCR F1=0.63 vs Donut-FT F1=0.78

### 7.5. Почему Donut-FT более робастен к искажениям

- **OCR**: каждая стадия деградирует независимо при corruption (blur ломает detection, JPEG artifacts ломают recognition, garbled text ломает extraction)
- **Donut**: Swin Transformer относительно устойчив к низкочастотному шуму; BART-декодер компенсирует повреждённые регионы как языковая модель
- **Наш результат**: OCR degradation -0.166 vs Donut-FT degradation -0.119

### 7.6. Почему fine-tuning критически важен для Donut

| Модель | F1 (наш) | F1 (опубликованный) |
|--------|:--------:|:-------------------:|
| Donut pre-trained | 0.02 | ~84% |
| Donut fine-tuned | 0.75 | ~94% |

Причины: (1) несовпадение формата выхода (SynthDoG schema ≠ SROIE tags), (2) domain gap (синтетические документы ≠ реальные малайзийские чеки), (3) декодер не знает специальные токены `<s_company>`, `<s_date>`, `<s_total>`.

---

## 8. Позиционирование наших результатов

### Разрыв с опубликованными SOTA

| Фактор | Влияние | Наша установка | Опубликованная SOTA |
|--------|:-------:|----------------|---------------------|
| Данные обучения | High | 240 docs (quick mode) | 626 docs (full) |
| Валидация | High | 80 docs (23%) | 347 docs (full) |
| Эпохи обучения | High | 2 эпохи | 30+ эпох |
| Layout-фичи | Medium | Нет | Text + 2D bbox + image |
| OCR-экстрактор | Medium | Простые regex-правила | Обученный NER |
| Разрешение | Low | 1280x960 (T4) | 2560x1920 |

### Ценность работы

Данный проект — **сравнительный анализ парадигм** (OCR vs end-to-end), а не попытка достичь SOTA. Ценность в:
1. Понимании **почему** разные подходы ведут себя по-разному на разных полях
2. Оценке робастности к реалистичным искажениям (messenger corruption)
3. Демонстрации критической важности fine-tuning для end-to-end моделей

---

## 9. Глубокий анализ ошибок

Дополнительный анализ в ноутбуке `06_deep_error_analysis.ipynb`:

### Таксономия ошибок
Каждая ошибка классифицирована по типу:
- **Vendor**: `empty_prediction`, `partial_match`, `address_confused`, `wrong_entity`, `hallucination`
- **Date**: `empty_prediction`, `wrong_year`, `wrong_month`, `wrong_day`, `format_error`
- **Total**: `empty_prediction`, `off_by_cents`, `wrong_amount`, `order_of_magnitude`

### Пересечение ошибок пайплайнов
Анализ показывает, что OCR и Donut-FT часто ошибаются на **разных** документах, что указывает на потенциал ансамблевого подхода (OCR для дат + Donut для vendor/total).

### Влияние суммы чека на точность
Графики показывают зависимость accuracy от диапазона суммы чека (<10, 10-50, 50-100, 100-500, 500+ RM).

---

## 10. Исправленные проблемы

В ходе отладки были решены следующие проблемы:

| Проблема | Решение |
|----------|---------|
| `ModuleNotFoundError: docai_utils` после рестарта ядра | Добавлен inline-fallback: `docai_utils.py` записывается прямо в рантайм, если Drive недоступен |
| CUDA OOM при fine-tuning на T4 | Уменьшение разрешения до 1280×960, fp16 mixed precision, освобождение VRAM от претрейн-моделей, batch=1+accum=2 |
| `CheckpointError` при gradient checkpointing + autocast | Отключён gradient checkpointing, оставлен только fp16 через `GradScaler` |
| EasyOCR `reader` удалён перед fine-tuning | Автоматическая переинициализация EasyOCR reader в ячейке robustness evaluation |
| Drive не смонтирован → `FileNotFoundError` | Fallback: полный код `docai_utils.py` встроен inline в ноутбук |

---

## 11. Выходные файлы

Все результаты сохранены в `v2/outputs/`:

| Файл | Описание |
|------|----------|
| `ocr_predictions_clean_quick.csv` | Предсказания OCR на чистых изображениях |
| `ocr_predictions_corrupted_quick.csv` | Предсказания OCR на искажённых изображениях |
| `donut_pretrained_predictions_quick.csv` | Предсказания Donut pretrained |
| `donut_predictions_clean_quick.csv` | Предсказания Donut fine-tuned (clean) |
| `donut_predictions_corrupted_quick.csv` | Предсказания Donut fine-tuned (corrupted) |
| `results_summary_quick.csv` | Сводная таблица всех метрик |
| `robustness_results_quick.csv` | Деградация F1 по полям |
| `fig_f1_by_field_quick.png` | График F1 по полям |
| `fig_degradation_quick.png` | График деградации при corruption |
| `fig_exact_match_quick.png` | График exact match по критичным полям |
| `fig_error_taxonomy_quick.png` | Таксономия ошибок: OCR vs Donut-FT |
| `fig_cross_pipeline_overlap_quick.png` | Пересечение ошибок пайплайнов |
| `fig_error_correlation_quick.png` | Корреляция ошибок по документам |
| `fig_accuracy_by_amount_quick.png` | Точность по диапазонам суммы чека |
| `fig_sroie_leaderboard_quick.png` | Наши результаты vs SROIE лидерборд |
| `fig_error_propagation_quick.png` | Каскадные ошибки: OCR vs End-to-end |

---

## 12. Ключевые выводы

1. **Donut-FT — лучший общий пайплайн** (Micro F1 = 0.75 vs 0.63 у OCR на clean данных)
2. **OCR Baseline выигрывает по Date** благодаря хорошим regex-правилам для дат
3. **Donut-FT более робастен**: средняя деградация F1 при messenger corruption +0.12 vs +0.17 у OCR
4. **Donut Pretrained (без дообучения) практически бесполезен** на SROIE (F1 ≈ 0) — требуется fine-tuning
5. **Финансово критичный Total** — Donut-FT значительно лучше по exact match (63.7% vs 46.2%)
6. **Ограничение**: quick mode (80 val docs, 2 эпохи) — для production-результатов нужен full mode (347 val docs, 5 эпох)
7. **Модели ошибаются на разных документах** — потенциал для гибридного/ансамблевого подхода
8. **Layout-информация** (не используется нами) добавляет 2-3 F1 пункта по данным литературы

---

## 13. Список литературы

1. Huang, Z., Chen, K., He, J., et al. (2019). "ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction." *ICDAR 2019.* arXiv:2103.10213
2. Kim, G., Hong, T., Yim, M., et al. (2022). "OCR-free Document Understanding Transformer." *ECCV 2022.* arXiv:2111.15664
3. Xu, Y., Li, M., Cui, L., et al. (2020). "LayoutLM: Pre-training of Text and Layout for Document Image Understanding." *KDD 2020.* arXiv:1912.13318
4. Xu, Y., Xu, Y., Lv, T., et al. (2021). "LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding." *ACL 2021.* arXiv:2012.14740
5. Huang, Y., Lv, T., Cui, L., et al. (2022). "LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking." *ACM MM 2022.* arXiv:2204.08387
6. Hong, T., Kim, D., Ji, M., et al. (2022). "BROS: A Pre-trained Language Model Focusing on Text and Layout for Better Key Information Extraction from Documents." *AAAI 2022.* arXiv:2108.04539
7. Yu, W., Lu, N., Qi, X., et al. (2021). "PICK: Processing Key Information Extraction from Documents using Improved Graph Learning-Convolutional Networks." *ICPR 2020.*
8. Schmid, P. (2022). "Fine-tuning Donut for document parsing." philschmid.de

---

## 14. Технологический стек

- **Python 3.12**, Google Colab (Tesla T4 GPU, 16 GB VRAM)
- **EasyOCR** — OCR baseline
- **Hugging Face Transformers** — Donut VisionEncoderDecoderModel
- **PyTorch 2.10** с fp16 mixed precision
- **OpenCV** — обработка изображений, corruption
- **pandas, numpy, matplotlib** — данные и визуализация

---

*Отчёт сгенерирован: 24 февраля 2026 г.*
