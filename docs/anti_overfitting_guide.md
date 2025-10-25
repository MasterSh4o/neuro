# Руководство по борьбе с переобучением модели

## 🎯 Проблема

Анализ метрик тренировки показал классическое переобучение после 4 эпохи:
- **Train loss**: продолжает падать (0.7025 → 0.5635)
- **Val F1-micro**: достигает пика на эпохе 4 (0.7657), затем ухудшается до 0.7161
- **Разрыв**: между тренировочными и валидационными метриками увеличивается

## 🛠️ Решения

### 1. Новая конфигурация (`configs/datasphere_anti_overfitting.yaml`)

**Регуляризация модели:**
- `stochastic_depth`: 0.2 → 0.3
- `dropout`: 0.15 → 0.25  
- `head_dropout`: 0.3 → 0.4

**Оптимизация обучения:**
- `lr`: 0.00065 → 0.0005 (уменьшен на 23%)
- `weight_decay`: 0.05 → 0.08 (увеличен на 60%)
- `early_stopping_patience`: 8 → 5 (более ранняя остановка)

**Scheduler:**
- `warmup_epochs`: 3 → 5 (плавнее начало)
- `factor`: 0.5 → 0.7 (более агрессивное снижение)
- `patience`: 2 → 1 (быстрая реакция на ухудшение)
- `min_lr`: 1e-6 → 5e-7

**Регуляризация loss:**
- `label_smoothing`: 0.05 → 0.1

### 2. Усиленные аугментации

**Базовые аугментации (сохранены):**
- Horizontal/Vertical flip (p=0.5)
- Rotate90 (p=0.5)

**Новые аугментации:**
- `random_rotation`: до ±15° (p=0.3)
- `random_crop`: масштаб 0.9-1.0 (p=0.3)
- `color_jitter`: яркость/контраст 0.1 (p=0.2)
- `gaussian_blur`: kernel=3 (p=0.2)
- `mixup`: α=0.2 (p=0.2)

### 3. Mixup регуляризация

Добавлена реализация Mixup в [`src/train.py`](src/train.py:60):
- Случайное смешивание двух примеров
- Вероятность применения: 20%
- Параметр α: 0.2

## 📊 Мониторинг переобучения

### Анализ текущей тренировки

```bash
python scripts/analyze_overfitting.py
```

Создает:
- `reports/overfitting_analysis.txt` - текстовый отчет
- `reports/overfitting_curves.png` - графики метрик

### Универсальный монитор

```bash
python src/utils/overfitting_monitor.py metrics/history_20251024-171427.json \
    --save-plots reports/curves.png \
    --save-report reports/analysis.txt
```

Автоматически определяет:
- Точку начала переобучения
- Разрыв между train/val метриками
- Генерирует персональные рекомендации

## 🚀 Запуск улучшенной тренировки

```bash
# Использование новой конфигурации
bash scripts/run_anti_overfitting.sh

# Или напрямую
python src/train.py --config configs/datasphere_anti_overfitting.yaml
```

## 📈 Ожидаемые результаты

1. **Более плавные кривые обучения** за счет усиленных аугментаций
2. **Снижение переобучения** благодаря увеличенной регуляризации
3. **Ранняя остановка** при ухудшении валидационных метрик
4. **Лучшее обобщение** на тестовых данных

## 🔧 Дополнительные рекомендации

### Если переобучение сохраняется:

1. **Увеличьте dropout** до 0.3-0.4
2. **Добавьте больше аугментаций**:
   ```yaml
   cutmix:
     enabled: true
     alpha: 1.0
     prob: 0.3
   ```
3. **Используйте Label Smoothing** 0.15-0.2
4. **Уменьшите learning rate** еще на 20-30%

### Если модель недоучивается:

1. **Уменьшите регуляризацию**:
   - `dropout`: 0.2 → 0.15
   - `weight_decay`: 0.08 → 0.06
2. **Увеличьте learning rate** на 20-30%
3. **Увеличьте patience** в scheduler

## 📋 Ключевые файлы

- [`configs/datasphere_anti_overfitting.yaml`](configs/datasphere_anti_overfitting.yaml) - новая конфигурация
- [`src/train.py`](src/train.py) - обновлен с Mixup
- [`src/data/dataset.py`](src/data/dataset.py) - усиленные аугментации
- [`src/utils/overfitting_monitor.py`](src/utils/overfitting_monitor.py) - мониторинг
- [`scripts/run_anti_overfitting.sh`](scripts/run_anti_overfitting.sh) - запуск

## ⚡ Быстрые команды

```bash
# Анализ текущей ситуации
python scripts/analyze_overfitting.py

# Запуск с защитой от переобучения  
bash scripts/run_anti_overfitting.sh

# Мониторинг новой тренировки
python src/utils/overfitting_monitor.py runs/metrics/history_YYYYMMDD-HHMMSS.json