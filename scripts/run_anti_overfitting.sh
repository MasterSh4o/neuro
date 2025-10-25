#!/bin/bash

# Скрипт для запуска тренировки с защитой от переобучения

echo "Запуск тренировки с конфигурацией против переобучения..."

# Устанавливаем переменные окружения
export CUDA_VISIBLE_DEVICES=0

# Запускаем тренировку с новой конфигурацией
python src/train.py --config configs/datasphere_anti_overfitting.yaml

echo "Тренировка завершена!"