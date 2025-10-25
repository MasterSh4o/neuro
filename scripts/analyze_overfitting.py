#!/usr/bin/env python3
"""
Скрипт для анализа переобучения на основе существующих метрик
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.overfitting_monitor import OverfittingMonitor

def main():
    # Анализируем существующую историю тренировки
    history_path = "metrics/history_20251024-171427.json"
    
    print("🔍 Анализ переобучения модели...")
    print(f"📁 Файл метрик: {history_path}")
    
    try:
        monitor = OverfittingMonitor(history_path)
        
        # Генерируем и выводим отчет
        report = monitor.generate_report("reports/overfitting_analysis.txt")
        print(report)
        
        # Сохраняем графики
        monitor.plot_training_curves("reports/overfitting_curves.png")
        
        print("\n📊 Результаты анализа сохранены в папке reports/")
        
    except FileNotFoundError:
        print(f"❌ Файл метрик не найден: {history_path}")
        print("Убедитесь, что путь к файлу истории тренировки корректен")
    except Exception as e:
        print(f"❌ Ошибка при анализе: {e}")

if __name__ == "__main__":
    main()