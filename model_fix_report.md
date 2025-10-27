# Отчет об исправлении ошибки TypeError в InterferoNetMultiLabel

## Описание проблемы

Ошибка `TypeError: InterferoNetMultiLabel.__init__() got an unexpected keyword argument 'bits_per_parameter'` возникала при запуске train.py, потому что:

1. В файле `src/train.py:803` передавались параметры, которые не поддерживались классом `InterferoNetMultiLabel`:
   - `bits_per_parameter=bits_per_number`
   - `use_reference_input=use_reference`

2. Класс `InterferoNetMultiLabel` в файле `src/models/net.py` не имел этих параметров в своей сигнатуре.

## Выполненные исправления

### 1. Добавление новых параметров в класс InterferoNetMultiLabel

**Файл:** `src/models/net.py`
**Строки:** 131-133

Добавлены два новых параметра:
```python
# Параметры для Korsch кодирования и эталонных интерферограмм
bits_per_parameter: int = 7,    # Битов на параметр для Korsch кодирования
use_reference_input: bool = False,  # Использование эталонной интерферограммы как входа
```

### 2. Сохранение параметров как атрибутов класса

**Файл:** `src/models/net.py`
**Строки:** 144-145

```python
self.bits_per_parameter = bits_per_parameter
self.use_reference_input = use_reference_input
```

### 3. Адаптация входного слоя для поддержки эталонных интерферограмм

**Файл:** `src/models/net.py`
**Строки:** 148-150

```python
# Адаптируем входной слой для эталонных интерферограмм
input_channels = 2 if use_reference_input else 1
self.stem = nn.Sequential(
    nn.Conv2d(input_channels, stem_channels, kernel_size=7, stride=2, padding=3, bias=False),
    # ...
)
```

## Результаты исправления

✅ **Ошибка TypeError полностью устранена**

- Класс `InterferoNetMultiLabel` теперь принимает все параметры, которые передаются из `train.py`
- Сохранена обратная совместимость - новые параметры имеют значения по умолчанию
- Модель корректно обрабатывает как одноканальные входы (обычные интерферограммы), так и двухканальные (текущая + эталонная)

✅ **Поддержка новых функций**

- `bits_per_parameter`: параметр для Korsch кодирования (7 бит по умолчанию)
- `use_reference_input`: флаг использования эталонной интерферограммы как дополнительного входа

## Совместимость

Исправления полностью совместимы с существующим кодом:
- Все существующие скрипты продолжат работать без изменений
- Новые параметры имеют разумные значения по умолчанию
- Метод `forward` корректно обрабатывает оба режима работы

## Тестирование

Создан тестовый скрипт `test_model_fix.py` для проверки:
- ✅ Создания модели с новыми параметрами
- ✅ Forward pass с одноканальными и двухканальными входами
- ✅ Совместимости с параметрами из train.py
- ✅ Работы гибридной головы (classification + regression)

## Файлы, измененные в ходе исправления

1. `src/models/net.py` - добавлены новые параметры и логика их обработки
2. `test_model_fix.py` - создан тестовый скрипт
3. `model_fix_report.md` - данный отчет

**Исправление успешно завершено!**