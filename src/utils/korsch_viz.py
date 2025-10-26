"""
Korsch Objective Visualization Tools

Визуализация смещений зеркал с субмикронной точностью.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, Any, Optional, Tuple, List
import torch

from .korsch_decoder import KorschDisplacementDecoder


class KorschVisualizer:
    """
    Визуализатор для анализа смещений трехзеркального объектива Корша
    с фокусом на субмикронной точности.
    """

    def __init__(
        self,
        decoder: Optional[KorschDisplacementDecoder] = None,
        figsize: Tuple[int, int] = (15, 10),
        dpi: int = 150,
        style: str = "seaborn-v0_8"
    ):
        """
        Args:
            decoder: Декодер для преобразования бит в физические единицы
            figsize: Размер фигур
            dpi: Разрешение изображений
            style: Стиль matplotlib
        """
        self.decoder = decoder
        self.figsize = figsize
        self.dpi = dpi
        self.style = style

        # Установка стиля
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')

        # Цветовая схема
        self.colors = {
            'mirror1': '#1f77b4',  # Синий
            'mirror2': '#ff7f0e',  # Оранжевый
            'predicted': '#2ca02c',  # Зеленый
            'target': '#d62728',    # Красный
            'error': '#9467bd',     # Фиолетовый
            'background': '#f8f9fa'  # Светлый фон
        }

    def plot_displacement_comparison(
        self,
        predicted_displacements: Dict[str, Any],
        target_displacements: Dict[str, Any],
        title: str = "Mirror Displacement Comparison",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Визуализация сравнения предсказанных и целевых смещений.
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # Параметры для визуализации
        param_types = [
            ('linear_x', 'Linear X', 'μm'),
            ('linear_y', 'Linear Y', 'μm'),
            ('linear_z', 'Linear Z', 'μm'),
            ('angular_x', 'Angular X', '"')
        ]
        param_types.append(('angular_y', 'Angular Y', '"'))

        ax_flat = axes.flatten()
        for i, (param_key, param_name, unit) in enumerate(param_types):
            if i >= len(ax_flat):
                break

            ax = ax_flat[i]

            # Сбор данных для всех зеркал
            mirrors_data = []
            for mirror_name in sorted(predicted_displacements.keys()):
                pred_val = predicted_displacements[mirror_name][param_key]['fine_value'] \
                    if 'fine_value' in predicted_displacements[mirror_name][param_key] \
                    else predicted_displacements[mirror_name][param_key]['value']

                target_val = target_displacements[mirror_name][param_key]['fine_value'] \
                    if 'fine_value' in target_displacements[mirror_name][param_key] \
                    else target_displacements[mirror_name][param_key]['value']

                error = abs(pred_val - target_val)

                mirrors_data.append({
                    'mirror': mirror_name,
                    'predicted': pred_val,
                    'target': target_val,
                    'error': error
                })

            # Построение графика
            x_pos = np.arange(len(mirrors_data))
            width = 0.35

            bars1 = ax.bar(x_pos - width/2, [d['predicted'] for d in mirrors_data],
                          width, label='Predicted', color=self.colors['predicted'], alpha=0.8)
            bars2 = ax.bar(x_pos + width/2, [d['target'] for d in mirrors_data],
                          width, label='Target', color=self.colors['target'], alpha=0.8)

            # Добавление значений ошибок
            for j, (bar1, bar2, data) in enumerate(zip(bars1, bars2, mirrors_data)):
                ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.01*max(abs(bar1.get_height()), abs(bar2.get_height())),
                       f'{data["error"]:.2f}', ha='center', va='bottom', fontsize=9)

            ax.set_xlabel('Mirror')
            ax.set_ylabel(f'{param_name} ({unit})')
            ax.set_title(f'{param_name} Displacement')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([d['mirror'].replace('_', ' ').title() for d in mirrors_data])
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Пороговые линии для субмикронной точности
            if 'μm' in unit:
                ax.axhline(y=1.0, color=self.colors['error'], linestyle='--', alpha=0.7, label='1 μm threshold')
                ax.axhline(y=-1.0, color=self.colors['error'], linestyle='--', alpha=0.7)
            else:
                ax.axhline(y=1.0, color=self.colors['error'], linestyle='--', alpha=0.7, label='1" threshold')
                ax.axhline(y=-1.0, color=self.colors['error'], linestyle='--', alpha=0.7)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_error_distribution(
        self,
        errors_data: List[Dict[str, float]],
        title: str = "Displacement Error Distribution",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Визуализация распределения ошибок с фокусом на субмикронных диапазонах.
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # Извлечение ошибок по типам
        error_types = {
            'linear_x': [],
            'linear_y': [],
            'linear_z': [],
            'angular_x': [],
            'angular_y': []
        }

        for error_dict in errors_data:
            for key, value in error_dict.items():
                if key in error_types:
                    error_types[key].append(value)

        # Визуализация распределений ошибок
        param_configs = [
            ('linear_x', 'Linear X (μm)', [0, 5], self.colors['mirror1']),
            ('linear_y', 'Linear Y (μm)', [0, 5], self.colors['mirror2']),
            ('linear_z', 'Linear Z (μm)', [0, 5], self.colors['predicted']),
            ('angular_x', 'Angular X (")', [0, 5], self.colors['target'])
        ]
        param_configs.append(('angular_y', 'Angular Y (")', [0, 5], self.colors['error']))

        for i, (param_key, title, x_range, color) in enumerate(param_configs):
            if i >= len(axes.flat):
                break

            ax = axes.flat[i]
            errors = error_types[param_key]

            if errors:
                # Гистограмма с KDE
                sns.histplot(errors, bins=30, kde=True, color=color, alpha=0.7, ax=ax)

                # Вертикальные линии порогов
                ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='1 unit threshold')
                ax.axvline(x=0.5, color='orange', linestyle='--', alpha=0.6, linewidth=1.5, label='0.5 unit threshold')

                # Статистика
                mean_error = np.mean(errors)
                median_error = np.median(errors)
                sub_unit_accuracy = np.mean(np.array(errors) < 1.0) * 100

                stats_text = f'Mean: {mean_error:.3f}\nMedian: {median_error:.3f}\n<1 unit: {sub_unit_accuracy:.1f}%'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                ax.set_xlabel(f'Error ({title.split()[2]})')
                ax.set_ylabel('Frequency')
                ax.set_title(title)
                ax.set_xlim(x_range)
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                        horizontalalignment='center', verticalalignment='center', fontsize=14)
                ax.set_title(title)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_precision_analysis(
        self,
        precision_data: Dict[str, Any],
        title: str = "Precision Analysis",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Анализ достижимой точности с различными битностями.
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # Точность в зависимости от битности
        bits_range = range(5, 9)  # 5-8 бит
        linear_precisions = []
        angular_precisions = []

        for bits in bits_range:
            levels = 2 ** bits
            linear_prec = 1000.0 / levels  # мкм на шаг
            angular_prec = 60.0 / levels  # угл. секунд на шаг
            linear_precisions.append(linear_prec)
            angular_precisions.append(angular_prec)

        # График 1: Точность vs битность
        ax1 = axes[0, 0]
        ax1.plot(bits_range, linear_precisions, 'bo-', label='Linear (μm)', linewidth=2, markersize=8)
        ax1.plot(bits_range, angular_precisions, 'ro-', label='Angular (")', linewidth=2, markersize=8)
        ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Target: <1 unit')
        ax1.set_xlabel('Bits per Parameter')
        ax1.set_ylabel('Precision per Step')
        ax1.set_title('Precision vs Bit Depth')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # График 2: Процент точности <1 единицы
        ax2 = axes[0, 1]
        sub_micron_capable = [prec <= 1.0 for prec in linear_precisions]
        sub_arcsec_capable = [prec <= 1.0 for prec in angular_precisions]

        ax2.bar(bits_range, [int(x) for x in sub_micron_capable], alpha=0.7,
               color=self.colors['mirror1'], label='Linear <1μm')
        ax2.bar(bits_range, [int(x) for x in sub_arcsec_capable], alpha=0.7,
               color=self.colors['mirror2'], label='Angular <1"')
        ax2.set_xlabel('Bits per Parameter')
        ax2.set_ylabel('Capable (0/1)')
        ax2.set_title('Sub-Micron/Sub-Arcsec Capability')
        ax2.set_ylim(-0.1, 1.1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # График 3: Текущая статистика
        ax3 = axes[1, 0]
        if self.decoder:
            stats = self.decoder.get_precision_stats()

            metric_names = ['Linear Precision', 'Angular Precision']
            metric_values = [stats['linear_precision_um'], stats['angular_precision_arcsec']]
            colors = [self.colors['mirror1'], self.colors['mirror2']]

            bars = ax3.bar(metric_names, metric_values, color=colors, alpha=0.8)

            # Добавление пороговой линии
            ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Target: <1 unit')

            # Значения на барах
            for bar, value in zip(bars, metric_values):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metric_values)*0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

            ax3.set_ylabel('Precision (units)')
            ax3.set_title('Current Decoder Precision')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # График 4: Память и вычислительные требования
        ax4 = axes[1, 1]
        memory_usage = [(2 ** bits) * 10 * 4 / 1024**2 for bits in bits_range]  # MB

        ax4_twin = ax4.twinx()

        line1 = ax4.plot(bits_range, memory_usage, 'g-o', label='Memory Usage', linewidth=2, markersize=8)
        line2 = ax4_twin.plot(bits_range, [2**bits for bits in bits_range], 'b-s',
                               label='Discretization Levels', linewidth=2, markersize=8)

        ax4.set_xlabel('Bits per Parameter')
        ax4.set_ylabel('Memory Usage (MB)', color='g')
        ax4_twin.set_ylabel('Discretization Levels', color='b')
        ax4.tick_params(axis='y', labelcolor='g')
        ax4_twin.tick_params(axis='y', labelcolor='b')
        ax4.set_title('Resource Requirements')

        # Комбинированная легенда
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper left')

        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_interferogram_with_displacements(
        self,
        interferogram: np.ndarray,
        displacements: Dict[str, Any],
        title: str = "Interferogram with Displacement Vectors",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Визуализация интерферограммы с наложенными векторами смещений.
        """
        fig, axes = plt.subplots(1, 2, figsize=(20, 10), dpi=self.dpi)
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # Левый subplot: интерферограмма
        ax1 = axes[0]
        im1 = ax1.imshow(interferogram, cmap='gray', extent=[0, interferogram.shape[1], 0, interferogram.shape[0]])
        ax1.set_title('Interferogram')
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        plt.colorbar(im1, ax=ax1, label='Intensity')

        # Правый subplot: интерферограмма с векторами смещений
        ax2 = axes[1]
        im2 = ax2.imshow(interferogram, cmap='gray', extent=[0, interferogram.shape[1], 0, interferogram.shape[0]])
        ax2.set_title('Interferogram with Displacement Vectors')
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')

        # Добавление векторов смещений
        img_height, img_width = interferogram.shape
        center_x, center_y = img_width // 2, img_height // 2

        mirror_positions = [
            (center_x - img_width//4, center_y),  # Mirror 1
            (center_x + img_width//4, center_y)   # Mirror 2
        ]

        for i, (mirror_name, mirror_data) in enumerate(displacements.items()):
            pos_x, pos_y = mirror_positions[i]

            # Масштабирование для визуализации
            scale_factor = 20  # Увеличиваем для видимости

            # Линейные смещения (вектор)
            linear_x = mirror_data.get('linear_x', {}).get('fine_value', 0) / scale_factor
            linear_y = mirror_data.get('linear_y', {}).get('fine_value', 0) / scale_factor

            # Угловые смещения (поворот)
            angular_x = mirror_data.get('angular_x', {}).get('fine_value', 0)
            angular_y = mirror_data.get('angular_y', {}).get('fine_value', 0)

            # Цвет для зеркала
            color = self.colors['mirror1'] if i == 0 else self.colors['mirror2']

            # Рисуем вектор линейного смещения
            ax2.arrow(pos_x, pos_y, linear_x, -linear_y,
                     head_width=15, head_length=10, fc=color, ec=color, alpha=0.8,
                     linewidth=2, label=f'{mirror_name} Linear')

            # Рисуем индикатор углового смещения
            if abs(angular_x) > 0.1 or abs(angular_y) > 0.1:
                circle = plt.Circle((pos_x, pos_y), radius=30, fill=False,
                                 edgecolor=color, linewidth=2, linestyle='--', alpha=0.8)
                ax2.add_patch(circle)
                ax2.text(pos_x, pos_y + 50, f'θx: {angular_x:.1f}"\nθy: {angular_y:.1f}"',
                        ha='center', va='bottom', color=color, fontweight='bold')

            # Маркер зеркала
            ax2.plot(pos_x, pos_y, 'o', color=color, markersize=10, markeredgecolor='white', markeredgewidth=2)
            ax2.text(pos_x, pos_y - 50, mirror_name.replace('_', ' ').title(),
                    ha='center', va='top', color=color, fontweight='bold', fontsize=12)

        # Легенда для масштаба
        scale_text = f'Scale: 1 unit = {scale_factor} pixels'
        ax2.text(0.02, 0.02, scale_text, transform=ax2.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig


def create_precision_report(
    decoder: KorschDisplacementDecoder,
    save_dir: str = "korch_precision_report"
) -> None:
    """
    Создает полный отчет по точности системы.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    # Анализ точности
    viz = KorschVisualizer(decoder)

    # График анализа точности
    fig1 = viz.plot_precision_analysis(
        decoder.get_precision_stats(),
        "Korsch Objective Precision Analysis"
    )
    fig1.savefig(f"{save_dir}/precision_analysis.png", dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # Сохранение статистики
    stats = decoder.get_precision_stats()

    with open(f"{save_dir}/precision_stats.txt", 'w', encoding='utf-8') as f:
        f.write("KORSCH OBJECTIVE PRECISION ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Bits per parameter: {stats['bits_per_parameter']}\n")
        f.write(f"  Discretization levels: {stats['discretization_levels']}\n")
        f.write(f"  Linear precision: {stats['linear_precision_um']:.3f} μm\n")
        f.write(f"  Angular precision: {stats['angular_precision_arcsec']:.3f} arcsec\n")
        f.write(f"  Linear range: ±{stats['linear_range_um']/2:.1f} μm\n")
        f.write(f"  Angular range: ±{stats['angular_range_arcsec']/2:.1f} arcsec\n\n")

        f.write("Capabilities:\n")
        f.write(f"  Sub-micron capable: {stats['sub_micron_capability']}\n")
        f.write(f"  Sub-arcsec capable: {stats['sub_arcsec_capability']}\n\n")

        # Рекомендации
        if not stats['sub_micron_capability']:
            recommended_bits = int(np.ceil(np.log2(2000.0)))  # Для <1 μm
            f.write(f"Recommendations:\n")
            f.write(f"  For <1 μm precision: use {recommended_bits} bits per parameter\n")
            f.write(f"  Current precision: {stats['linear_precision_um']:.3f} μm per step\n")
            f.write(f"  Required precision: <1.000 μm per step\n")

    print(f"Precision report saved to {save_dir}/")


if __name__ == "__main__":
    # Пример использования
    decoder = KorschDisplacementDecoder(bits_per_param=7, use_hybrid=True)
    create_precision_report(decoder)