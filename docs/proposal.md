# Поиск по фото с текстовыми инструкциями (Fashion-Finder)

**Название проекта:** Fashion-Finder — поиск по фото с текстовыми инструкциями
**ФИО:** Финенко Елизавета
**Курс:** МФТИ MLOps, весна 2026

---

## Постановка задачи

Проект Fashion-Finder посвящён задаче Composed Image Retrieval (CIR). Цель — поиск целевого изображения в каталоге одежды по комбинированному запросу: референсное изображение + текстовое описание-модификатор, указывающее, какие изменения необходимо применить к исходной картинке. Задача применима в e-commerce (поиск товаров «как это, но с длинными рукавами»), рекомендательных системах и системах визуального поиска.

## Формат входных и выходных данных

Протокол взаимодействия на этапе инференса:

**Входные данные:**

| Поле            | Формат                                                  | Размер / ограничения |
| --------------- | ------------------------------------------------------- | -------------------- |
| Reference image | RGB JPEG/PNG, resize до 256 × 256, центр-кроп 224 × 224 | ≤ 1 МБ               |
| Modifier text   | UTF-8 строка, ≤ 64 BPE токенов после truncation         | ≤ 200 символов       |

**Выходные данные:**

Упорядоченный список (top-K, по умолчанию K = 10) извлечённых изображений из предсохранённой галереи по убыванию релевантности. JSON-формат:

```json
{
  "query_image": "B009ZD7XLC.jpg",
  "query_text": "have longer sleeves and a darker color",
  "results": [
    { "rank": 1, "score": 0.83, "image": "B003GUI9MA.jpg" },
    { "rank": 2, "score": 0.81, "image": "B007NUA8X4.jpg" }
  ]
}
```

API: HTTP-сервис через Triton Inference Server (порт 8000) либо CLI-интерфейс (`infer.py query`).

**Пример пары query → target:**

| Reference (вход)                                              | Text modifier (вход)                       | Target (ожидаемый выход)                                      |
| ------------------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------- |
| Платье с короткими рукавами светлого цвета (`B009ZD7XLC.jpg`) | `"have longer sleeves and a darker color"` | Платье с длинными рукавами тёмного оттенка (`B003GUI9MA.jpg`) |

## Метрики

Основной класс метрик — **Recall@K**, так как задача сводится к Information Retrieval, и важно, чтобы релевантное целевое изображение оказалось в верхней части поисковой выдачи. Используются **5 метрик** с обоснованными целевыми значениями по бейзлайнам Fashion-IQ.

| Метрика     | Семантика                                                | Целевое значение | Источник бейзлайна                                                                                                                                    |
| ----------- | -------------------------------------------------------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Recall@1    | Целевая картинка на 1-м месте                            | ≥ 0.18           | [Wu et al. 2021, Fashion-IQ paper](https://arxiv.org/abs/1905.12794)                                                                                  |
| Recall@5    | Целевая картинка в top-5                                 | ≥ 0.30           | [Wu et al. 2021](https://arxiv.org/abs/1905.12794)                                                                                                    |
| Recall@10   | Целевая картинка в top-10                                | ≥ 0.40           | TIRG [Vo et al. 2019] R@10 = 17.7%; CIRPLANT [Liu et al. 2021] R@10 = 18.5%; ARTEMIS [Delmas et al. 2022] R@10 = 25.0%. Наша цель — превзойти ARTEMIS |
| Recall@50   | Целевая картинка в top-50                                | ≥ 0.60           | ARTEMIS R@50 ≈ 47%; CoLLM [Huynh et al. 2025] R@50 ≈ 65%                                                                                              |
| Mean Recall | Среднее R@10 и R@50 по категориям dress / shirt / toptee | ≥ 0.50           | Используется как primary metric для отбора чекпоинтов                                                                                                 |

Текущая reference-реализация на основе CoLLM-архитектуры даёт avg R@10 ≈ 0.39 и R@50 ≈ 0.61 на валидационном сплите Fashion-IQ, что подтверждает реалистичность поставленных целей.

## Валидация и тест

- Разделение на train / val / test производится в строгом соответствии с **официальными сплитами** датасета Fashion-IQ (файлы `cap.{dress,shirt,toptee}.{train,val,test}.json` из репозитория [XiaoxiaoGuo/fashion-iq](https://github.com/XiaoxiaoGuo/fashion-iq)).
- MT-CIR используется только на стадии pretraining (без val/test сплита).
- `random_seed = 42` фиксируется через `pl.seed_everything(workers=True)`.
- Управление зависимостями — `uv` + `pyproject.toml` + `uv.lock` для воспроизводимости окружения.
- Версия кода логируется в MLflow как hyperparam `git_commit` (SHA текущего HEAD).
- Все hyperparameters и итоговый resolved-config записываются в директорию запуска (`outputs/<date>/<time>/resolved_config.yaml`).

## Датасеты

### MT-CIR — для pretraining

| Параметр        | Значение                                                                 |
| --------------- | ------------------------------------------------------------------------ |
| Источник        | https://huggingface.co/datasets/chuonghm/MT-CIR                          |
| Авторы          | Hoang et al. (Chuonghm на HuggingFace)                                   |
| Дата публикации | 2024                                                                     |
| Объём           | ≈ 3.5 млн пар (reference image + modifier + target image)                |
| Размер на диске | ~ 250 ГБ (включая изображения); каталог аннотаций `mtcir.jsonl` — 995 МБ |
| Использование   | Только pretraining, без val/test                                         |

### Fashion-IQ — для finetuning и валидации

| Параметр        | Значение                                                          |
| --------------- | ----------------------------------------------------------------- |
| Источник        | https://github.com/XiaoxiaoGuo/fashion-iq                         |
| Авторы          | Hui Wu, Yupeng Gao, Xiaoxiao Guo et al. (IBM Research)            |
| Дата публикации | 2019 (CVPR-W 2020)                                                |
| Объём           | 77 684 изображения, 30 134 пары с текстовыми инструкциями         |
| Размер на диске | ~ 12 ГБ (полный набор изображений)                                |
| Категории       | dress, shirt, toptee                                              |
| Использование   | Finetuning train-сплит + валидация по val-сплиту каждой категории |

### Особенности данных и сложности

- **Шумность аннотаций в MT-CIR:** модификаторы сгенерированы автоматически (VLM-разметка), часть из них неинформативна или избыточна. Компенсируется большим объёмом + word-dropout аугментацией текста на тренировке.
- **Дисбаланс категорий в Fashion-IQ:** распределение пар по dress / shirt / toptee неравномерно (~10k / ~10k / ~10k, но плотность галереи отличается). Валидируем каждую категорию отдельно и усредняем через Mean Recall.
- **Неоднозначность модификаторов:** одна пара (ref, target) имеет два независимых caption-а, написанных разными аннотаторами, что отражает субъективность задачи. На валидации обрабатываются оба caption-а с независимым подсчётом метрик.
- **Размер vision-language модели:** базовая CoLLM использует LLM семейства Mistral/Salesforce SFR (7B параметров). Полное обучение требует ≥ 24 ГБ VRAM, что вынуждает использовать LoRA-адаптеры, bf16-mixed precision и gradient checkpointing.

## Моделирование

### Бейзлайн — CLIP zero-shot

Используется как точка отсчёта, не входит в основной репозиторий обучения.

**Стадии пайплайна:**

1. **Препроцессинг текста:** CLIP-токенайзер (BPE), padding до 77 токенов.
2. **Препроцессинг изображения:** resize 224 × 224, нормализация ImageNet-mean/std.
3. **Forward:** CLIP ViT-B/16 даёт два эмбеддинга — text_emb и image_emb (по 512-dim каждый).
4. **Композиция запроса:** простое сложение `query_emb = image_emb + text_emb`, L2-нормализация.
5. **Поиск:** FAISS HNSW32 kNN-search по галерее target-изображений, закодированных тем же CLIP.

Ожидаемое качество — R@10 в районе 5–10% (не дообучено под domain fashion).

### Основная модель — CoLLM

Реализация по статье [CoLLM: A Large Language Model for Composed Image Retrieval (Huynh et al., 2025)](https://arxiv.org/abs/2503.19910), фреймворк PyTorch + PyTorch Lightning. Архитектура слияния модальностей через LLM, обучаемая контрастивно.

**Архитектурные блоки:**

1. **Vision Encoder f(·):** ViT (timm `vit_base_patch16_clip_224.openai` для production, `vit_tiny_patch16_224` для smoke-режима на CPU). Извлекает CLS-токен → vision_features размерности 768 / 192.
2. **Image Adapter g(·):** двухслойный MLP (Linear → GELU → Linear), проецирующий vision_features в hidden-размерность LLM.
3. **LLM Φ(·):** `Salesforce/SFR-Embedding-2_R` (Mistral-7B base) для production либо `HuggingFaceTB/SmolLM2-135M` для smoke. LoRA-адаптеры (r=16, target_modules=`q_proj, k_proj, v_proj, o_proj`) — обучаемая часть, базовые веса заморожены. Gradient checkpointing активен для экономии памяти.
4. **Projection heads:** `llm_proj` (Linear → 4096-dim для composed запроса), `target_proj` (Linear → 4096-dim для target image).
5. **Logit scale:** обучаемый параметр температуры softmax, инициализация `log(1/0.07) ≈ 2.659`.

**Стадии пайплайна:**

1. **Препроцессинг изображения:** PIL → RGB → resize → центр-кроп 224 × 224 → ToTensor → нормализация (mean=0.5, std=0.5).
2. **Препроцессинг текста:** `AutoTokenizer` LLM + специальный токен `<image>`, prompt-шаблон `Instruct: Find the image that matches the query.\nQuery:\nImage: <image>\nText: {modifier}`, padding до 64 токенов.
3. **Forward composed query:**
   - vision_features = `f(reference_image)`
   - image_token = `g(vision_features)` → подмешивается в input_embeds LLM на позиции `<image>`-токена
   - hidden_states = LLM forward
   - composed_emb = `llm_proj(hidden_states[-1])` — берётся скрытое состояние последнего ненулевого токена
4. **Forward target:** target_emb = `target_proj(f(target_image))`
5. **Loss:** двусторонняя InfoNCE (Batch-Centric Contrastive Loss) — cross-entropy на матрице сходств `composed_emb @ target_emb.T * exp(logit_scale)`, симметрично i→t и t→i.
6. **Постпроцессинг:** L2-нормализация эмбеддингов, FAISS HNSW32 индекс по галерее.

**Стадии обучения:**

| Этап     | Датасет          | Эпохи | Замороженные части                            | LR   | Прецизион  |
| -------- | ---------------- | ----- | --------------------------------------------- | ---- | ---------- |
| Pretrain | MT-CIR           | 3     | Vision encoder                                | 1e-4 | bf16-mixed |
| Finetune | Fashion-IQ train | 10    | Vision encoder (LoRA в LLM активна с эпохи 0) | 1e-4 | bf16-mixed |

## Внедрение

### Production-формат модели

**Обязательный формат:** ONNX opset 17 для онлайн-инференса. Дополнительно — TensorRT plan (FP16) для batched-инференса под GPU.

**Артефакты поставки:**

- `fashion_finder_vision.onnx` (≈ 700 МБ) — кодирование элементов галереи (target images)
- `fashion_finder_composer.onnx` (≈ 7 ГБ) — кодирование composed запроса (image + text)
- `gallery.faiss` + `gallery.manifest.json` — индекс HNSW32, ~ 2 ГБ на 50 000 изображений
- `tokenizer/` — токенайзер LLM из HuggingFace

### Ресурсы для инференса

| Окружение                        | CPU    | RAM                    | Latency / запрос | Throughput               |
| -------------------------------- | ------ | ---------------------- | ---------------- | ------------------------ |
| Inference CPU (ONNX Runtime)     | 8 vCPU | 16 ГБ                  | ~ 250 мс         | ~ 30 QPS на одной машине |
| Inference GPU T4 (TensorRT FP16) | —      | 12 ГБ VRAM + 8 ГБ host | ~ 35 мс          | ~ 400 QPS при batch 16   |

### Пайплайн инференса (production)

```
HTTP POST /v2/models/fashion_finder_composer/infer
    ├─ Image tensor (1, 3, 224, 224) FP32
    ├─ input_ids tensor (1, 64) INT64
    └─ attention_mask tensor (1, 64) INT64
        │
        ▼
Triton Inference Server (dynamic batching, GPU)
        │
        ▼
4096-dim composed embedding (L2-normalized)
        │
        ▼
FAISS HNSW32 kNN search (ef_search=64)
        │
        ▼
Top-K target image IDs
        │
        ▼
JSON response: { "results": [{rank, score, image}, ...] }
```

### Инфраструктурная обвязка

- **Управление зависимостями:** `uv` + `pyproject.toml` + `uv.lock`
- **Изоляция среды:** виртуальное окружение через `uv` (Python 3.10–3.12), опционально Docker для production
- **Контроль качества кода:** pre-commit хуки (ruff lint + format, prettier для YAML / Markdown, базовые pre-commit-hooks)
- **Версионирование кода:** Git, GitHub репозиторий, main-ветка
- **Версионирование данных и моделей:** DVC с двумя локальными хранилищами (`.dvc/storage_data` для датасетов, `.dvc/storage_models` для чекпоинтов)
- **Конфигурация:** Hydra иерархические YAML-конфиги (`configs/data/`, `configs/model/`, `configs/trainer/`, `configs/logging/`, `configs/inference/`) с единой точкой входа `configs/config.yaml`
- **Логирование экспериментов:** MLflow (UI на 127.0.0.1:8080) + локальный TensorBoard в директории запуска
- **Inference сервер:** Triton Inference Server (NVIDIA NGC tritonserver:24.05-py3) с двумя моделями в `model_repository/` + Python test-client
- **Оптимизация инференса:** ONNX export + TensorRT plan (через `trtexec` shell-скрипт)
- **CLI-интерфейс:** `fashion-finder` через `fire` (entry-point `pyproject.toml`); основные команды — `download-data`, `pretrain`, `finetune`, `export-onnx`, `dvc-pull`

Репозиторий: https://github.com/filzzzy/fashion-finder
