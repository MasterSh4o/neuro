"""
Утилиты для мониторинга и анализа переобучения
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Optional


class OverfittingMonitor:
    """Класс для анализа переобучения на основе истории тренировки"""
    
    def __init__(self, history_path: str):
        self.history_path = Path(history_path)
        self.history = self._load_history()
        
    def _load_history(self) -> List[Dict[str, Any]]:
        """Загрузка истории тренировки из JSON файла"""
        if not self.history_path.exists():
            raise FileNotFoundError(f"History file not found: {self.history_path}")
        
        with open(self.history_path, 'r') as f:
            return json.load(f)
    
    def detect_overfitting_point(self, window_size: int = 3, patience: int = 2) -> Optional[int]:
        """
        Определяет точку начала переобучения
        
        Args:
            window_size: размер скользящего окна для сглаживания
            patience: количество эпох ухудшения перед определением переобучения
            
        Returns:
            Номер эпохи, с которой начинается переобучение, или None если переобучения нет
        """
        if len(self.history) < window_size + patience:
            return None
        
        train_losses = [epoch['train_loss'] for epoch in self.history]
        val_f1_scores = [epoch['val']['f1_micro'] for epoch in self.history]
        
        # Сглаживаем метрики скользящим средним
        smooth_train = self._moving_average(train_losses, window_size)
        smooth_val_f1 = self._moving_average(val_f1_scores, window_size)
        
        # Ищем точку, где валидационная метрика начинает ухудшаться
        # а тренировочная продолжает улучшаться
        best_val_epoch = 0
        best_val_score = smooth_val_f1[0]
        
        for i in range(1, len(smooth_val_f1)):
            if smooth_val_f1[i] > best_val_score:
                best_val_score = smooth_val_f1[i]
                best_val_epoch = i
            elif smooth_val_f1[i] < best_val_score - 0.001:  # порог ухудшения
                # Проверяем, что тренировочный loss продолжает падать
                if i > 0 and smooth_train[i] < smooth_train[best_val_epoch]:
                    return i + 1  # возвращаем номер эпохи (1-based)
        
        return None
    
    def _moving_average(self, data: List[float], window_size: int) -> List[float]:
        """Вычисление скользящего среднего"""
        if len(data) < window_size:
            return data
        
        result = []
        for i in range(len(data)):
            start = max(0, i - window_size + 1)
            window = data[start:i+1]
            result.append(np.mean(window))
        return result
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Построение графиков тренировочных кривых"""
        epochs = [epoch['epoch'] for epoch in self.history]
        train_losses = [epoch['train_loss'] for epoch in self.history]
        val_losses = [epoch['val']['loss'] for epoch in self.history]
        val_f1_scores = [epoch['val']['f1_micro'] for epoch in self.history]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # График loss
        ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # График F1-score
        ax2.plot(epochs, val_f1_scores, 'g-', label='Val F1-micro', linewidth=2)
        overfit_point = self.detect_overfitting_point()
        if overfit_point:
            ax2.axvline(x=overfit_point, color='red', linestyle='--', 
                       label=f'Overfitting starts at epoch {overfit_point}')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('F1-micro Score')
        ax2.set_title('Validation F1-micro Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Графики сохранены в: {save_path}")
        else:
            plt.show()
    
    def get_recommendations(self) -> List[str]:
        """Генерирует рекомендации по борьбе с переобучением"""
        recommendations = []
        overfit_point = self.detect_overfitting_point()
        
        if overfit_point is None:
            recommendations.append("✅ Признаков переобучения не обнаружено")
            return recommendations
        
        recommendations.append(f"⚠️  Переобучение начинается с эпохи {overfit_point}")
        
        # Анализируем параметры тренировки
        train_losses = [epoch['train_loss'] for epoch in self.history]
        val_losses = [epoch['val']['loss'] for epoch in self.history]
        
        # Разрыв между тренировочным и валидационным loss
        final_gap = val_losses[-1] - train_losses[-1]
        if final_gap > 0.1:
            recommendations.append("🔧 Большой разрыв между train и val loss - увеличьте регуляризацию")
            recommendations.append("   - Увеличьте dropout (0.2 → 0.3)")
            recommendations.append("   - Увеличьте weight decay (0.05 → 0.08)")
            recommendations.append("   - Добавьте L2 регуляризацию")
        
        # Анализ скорости обучения
        lrs = [epoch['lr'] for epoch in self.history]
        if lrs[-1] < lrs[0] * 0.1:
            recommendations.append("🔧 Learning rate слишком сильно уменьшился - используйте scheduler с более высоким min_lr")
        
        # Рекомендации по early stopping
        recommendations.append(f"🔧 Используйте early stopping с patience={min(3, overfit_point-1)}")
        
        # Рекомендации по аугментациям
        recommendations.append("🔧 Усильте аугментации данных:")
        recommendations.append("   - Увеличьте вероятность геометрических преобразований")
        recommendations.append("   - Добавьте случайный шум и размытие")
        recommendations.append("   - Используйте Mixup или CutMix")
        
        return recommendations
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """Генерирует полный отчет о переобучении"""
        report = []
        report.append("=" * 60)
        report.append("АНАЛИЗ ПЕРЕОБУЧЕНИЯ МОДЕЛИ")
        report.append("=" * 60)
        
        # Общая информация
        report.append(f"Файл истории: {self.history_path}")
        report.append(f"Всего эпох: {len(self.history)}")
        
        # Лучшие метрики
        best_epoch = max(self.history, key=lambda x: x['val']['f1_micro'])
        report.append(f"Лучшая эпоха: {best_epoch['epoch']} (F1-micro: {best_epoch['val']['f1_micro']:.4f})")
        
        # Анализ переобучения
        overfit_point = self.detect_overfitting_point()
        if overfit_point:
            report.append(f"🔴 Переобучение detected с эпохи {overfit_point}")
            if overfit_point <= best_epoch['epoch']:
                report.append("   ⚠️  Лучшая метрика достигнута после начала переобучения")
        else:
            report.append("✅ Переобучение не detected")
        
        report.append("\nРЕКОМЕНДАЦИИ:")
        report.append("-" * 40)
        for rec in self.get_recommendations():
            report.append(rec)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"Отчет сохранен в: {save_path}")
        
        return report_text


def main():
    """Пример использования"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Анализ переобучения модели")
    parser.add_argument("history_path", help="Путь к файлу истории тренировки")
    parser.add_argument("--save-plots", help="Путь для сохранения графиков")
    parser.add_argument("--save-report", help="Путь для сохранения отчета")
    
    args = parser.parse_args()
    
    monitor = OverfittingMonitor(args.history_path)
    
    # Генерируем отчет
    report = monitor.generate_report(args.save_report)
    print(report)
    
    # Строим графики
    monitor.plot_training_curves(args.save_plots)


if __name__ == "__main__":
    main()