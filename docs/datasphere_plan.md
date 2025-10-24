# План адаптации под Yandex DataSphere g2.1

## 0. Контекст и ограничения
- Профиль окружения: 28 vCPU, 1 GPU NVIDIA A100, хранение артефактов на /home/jupyter/work.
- Базовая точка входа обучения — [`main()`](src/train.py:495); необходимо сохранить совместимость с локальным запуском.
- Архитектура модели описана в [`InterferoNetMultiLabel`](src/models/net.py:103); текущий head требует переработки под новые скрытые слои.

## 1. Подготовка тренировочного окружения
- Добавить облачную конфигурацию [`configs/datasphere.yaml`](configs/datasphere.yaml) с путями, batch size, логированием и флагами AMP/channels_last.
- Обновить скрипт запуска в [`scripts/run_datasphere.sh`](scripts/run_datasphere.sh) для инициализации окружения DataSphere и передачи аргумента --config.
- Актуализировать обработку путей и генерацию директорий вывода в [`main()`](src/train.py:495), учитывая рабочую директорию DataSphere.

## 2. Оптимизация расписания и контроль ресурсов
- Расширить гибкость планировщика через [`WarmupReduceLROnPlateau`](src/utils/scheduler.py:33), разрешив выбор name=warmup_plateau в конфиге.
- Убедиться, что [`save_checkpoint()`](src/train.py:453) корректно сериализует состояние как CosineWarmupLR, так и WarmupReduceLROnPlateau.
- Настроить градиентный клиппинг и логирование пропусков шагов в обучении внутри цикла [`main()`](src/train.py:787).

## 3. Модификация архитектуры (три скрытых слоя)
- После глобального pooling в [`InterferoNetMultiLabel`](src/models/net.py:155) добавить head: Flatten -> Linear1 -> Activation -> Dropout -> Linear2 -> Activation -> Dropout -> Linear3 -> Activation -> Dropout -> Linear_out (50).
- Вынести размеры скрытых слоёв и вероятность dropout в конфигурацию через параметры model.hidden_dims и model.head_dropout.
- Обновить инициализацию весов в [`_init_weights`](src/models/net.py:164) для новых полносвязных слоёв.

## 4. Регуляризация и нормализация
- Оценить добавление LayerNorm или BatchNorm1d между скрытыми слоями head и параметризовать выбор через конфиг.
- Уточнить значения stochastic_depth и базового dropout в [`configs/local.yaml`](configs/local.yaml:14) и новом файле datasphere.

## 5. Функция потерь и label smoothing
- Завершить внедрение [`BCEWithLogitsLossWithSmoothing`](src/train.py:52) с конфигурируемыми параметрами label_smoothing и loss_reduction.
- Протоколировать активные настройки в обучении, чтобы логи DataSphere отображали значение ε и pos_weight.

## 6. Обновление конфигураций
- Расширить секцию train в [`configs/local.yaml`](configs/local.yaml:71) и [`configs/datasphere.yaml`](configs/datasphere.yaml) параметрами scheduler, label_smoothing, hidden_dims, head_dropout.
- Задать eval.threshold и testing.smoke_samples для ускоренной проверки в облаке.

## 7. Финальная проверка и документы
- Провести smoke-тест на подмножестве данных (конфиг testing.smoke_samples) и убедиться в корректности сохранения best.pt.
- Обновить README или отдельную инструкцию с шагами запуска в DataSphere, включая команды из scripts/run_datasphere.sh.
- После успешной проверки зафиксировать изменения (git commit) и синхронизировать чекпоинты при необходимости.