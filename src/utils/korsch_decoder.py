"""
Korsch Objective Displacement Decoder

Специализированные функции для декодирования бинарных меток
в физические единицы с микронной/угловой точностью.
"""

import numpy as np
import torch
from typing import Tuple, Dict, Any, Optional


class KorschDisplacementDecoder:
    """
    Декодер смещений трехзеркального объектива Корша.

    Поддерживает различные битности и гибридный подход
    (классификация + регрессия) для достижения <1 мкм точности.
    """

    def __init__(
        self,
        bits_per_param: int = 5,
        linear_range_um: float = 1000.0,  # ±1 мм в микронах
        angular_range_arcsec: float = 60.0,  # ±1 угловая минута в секундах
        use_hybrid: bool = False,
        regression_scale_factor: float = 2.0,  # Коэффициент масштабирования регрессии
    ):
        """
        Args:
            bits_per_param: Количество бит на параметр (5-8)
            linear_range_um: Диапазон линейных смещений в микронах (диаметр)
            angular_range_arcsec: Диапазон угловых смещений в угловых секундах (диаметр)
            use_hybrid: Использовать гибридный подход с регрессией
            regression_scale_factor: Масштаб регрессионных значений
        """
        self.bits_per_param = bits_per_param
        self.levels = 2 ** bits_per_param
        self.linear_range_um = linear_range_um
        self.angular_range_arcsec = angular_range_arcsec
        self.use_hybrid = use_hybrid
        self.regression_scale_factor = regression_scale_factor

        # Расчет шага дискретизации
        self.linear_step_um = linear_range_um / self.levels
        self.angular_step_arcsec = angular_range_arcsec / self.levels

        # Центр диапазона (нулевое смещение)
        self.linear_center_um = linear_range_um / 2
        self.angular_center_arcsec = angular_range_arcsec / 2

    def decode_classification_bits(
        self,
        binary_vector: np.ndarray | torch.Tensor,
        mirror_count: int = 2
    ) -> Dict[str, Any]:
        """
        Декодирование классификационных бит в физические единицы.

        Args:
            binary_vector: Вектор бинарных меток
            mirror_count: Количество зеркал (по умолчанию 2)

        Returns:
            Dict с параметрами смещений в микронах и угловых секундах
        """
        if isinstance(binary_vector, torch.Tensor):
            binary_vector = binary_vector.cpu().numpy()

        # Проверка размера
        expected_bits = mirror_count * 5 * self.bits_per_param  # 5 параметров на зеркало
        if len(binary_vector) != expected_bits:
            raise ValueError(
                f"Expected {expected_bits} bits for {mirror_count} mirrors with {self.bits_per_param} bits per parameter, "
                f"got {len(binary_vector)}"
            )

        displacements = {}

        for mirror in range(mirror_count):
            mirror_data = {}

            # Индексы параметров для текущего зеркала
            base_idx = mirror * 5 * self.bits_per_param

            # Декодирование 5 параметров зеркала
            # Параметры: [Угловой X, Угловой Y, Линейный X, Линейный Y, Линейный Z]
            param_names = ['angular_x', 'angular_y', 'linear_x', 'linear_y', 'linear_z']

            for i, param_name in enumerate(param_names):
                param_bits_idx = base_idx + i * self.bits_per_param
                param_bits = binary_vector[param_bits_idx:param_bits_idx + self.bits_per_param]

                # Преобразование бит в целое значение
                value = 0
                for j, bit in enumerate(reversed(param_bits)):
                    value |= (int(bit) << j)

                # Преобразование в физические единицы
                if param_name.startswith('angular'):
                    # Угловые параметры в угловых секундах
                    physical_value = (value - self.angular_center_arcsec / self.angular_step_arcsec) * self.angular_step_arcsec
                    unit = 'arcsec'
                else:
                    # Линейные параметры в микронах
                    physical_value = (value - self.linear_center_um / self.linear_step_um) * self.linear_step_um
                    unit = 'um'

                mirror_data[param_name] = {
                    'value': physical_value,
                    'unit': unit,
                    'discrete_level': value
                }

            displacements[f'mirror_{mirror + 1}'] = mirror_data

        return displacements

    def decode_hybrid_output(
        self,
        classification_bits: np.ndarray | torch.Tensor,
        regression_values: np.ndarray | torch.Tensor,
        mirror_count: int = 2
    ) -> Dict[str, Any]:
        """
        Декодирование гибридного выхода (классификация + регрессия).

        Args:
            classification_bits: Классификационные биты
            regression_values: Регрессионные значения для тонкой коррекции
            mirror_count: Количество зеркал

        Returns:
            Dict с параметрами смещений повышенной точности
        """
        if isinstance(classification_bits, torch.Tensor):
            classification_bits = classification_bits.cpu().numpy()
        if isinstance(regression_values, torch.Tensor):
            regression_values = regression_values.cpu().numpy()

        # Базовое декодирование классификации
        coarse_displacements = self.decode_classification_bits(classification_bits, mirror_count)

        # Применение регрессионной коррекции
        if len(regression_values) != mirror_count * 5:
            raise ValueError(
                f"Expected {mirror_count * 5} regression values, got {len(regression_values)}"
            )

        fine_displacements = {}

        reg_idx = 0
        for mirror_name in sorted(coarse_displacements.keys()):
            fine_mirror_data = {}
            coarse_data = coarse_displacements[mirror_name]

            param_names = ['angular_x', 'angular_y', 'linear_x', 'linear_y', 'linear_z']

            for i, param_name in enumerate(param_names):
                coarse_value = coarse_data[param_name]['value']
                unit = coarse_data[param_name]['unit']

                # Регрессионная коррекция
                regression_correction = regression_values[reg_idx]

                if unit == 'arcsec':
                    # Коррекция в угловых секундах
                    scale = self.angular_step_arcsec / self.regression_scale_factor
                else:
                    # Коррекция в микронах
                    scale = self.linear_step_um / self.regression_scale_factor

                fine_value = coarse_value + regression_correction * scale

                fine_mirror_data[param_name] = {
                    'coarse_value': coarse_value,
                    'regression_correction': regression_correction,
                    'fine_value': fine_value,
                    'unit': unit,
                    'correction_scale_um': scale if unit == 'um' else None,
                    'correction_scale_arcsec': scale if unit == 'arcsec' else None
                }

                reg_idx += 1

            fine_displacements[mirror_name] = fine_mirror_data

        return fine_displacements

    def get_precision_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику точности для текущих настроек.
        """
        return {
            'bits_per_parameter': self.bits_per_param,
            'discretization_levels': self.levels,
            'linear_precision_um': self.linear_step_um,
            'angular_precision_arcsec': self.angular_step_arcsec,
            'linear_range_um': self.linear_range_um,
            'angular_range_arcsec': self.angular_range_arcsec,
            'use_hybrid_mode': self.use_hybrid,
            'sub_micron_capability': self.linear_step_um < 1.0,
            'sub_arcsec_capability': self.angular_step_arcsec < 1.0
        }

    def validate_displacement_ranges(self, displacements: Dict[str, Any]) -> bool:
        """
        Проверяет, что смещения находятся в допустимых диапазонах.
        """
        for mirror_name, mirror_data in displacements.items():
            for param_name, param_info in mirror_data.items():
                if isinstance(param_info, dict):
                    value = param_info.get('fine_value', param_info.get('value', 0))
                    unit = param_info['unit']

                    if unit == 'um':
                        if abs(value) > self.linear_range_um / 2:
                            return False
                    elif unit == 'arcsec':
                        if abs(value) > self.angular_range_arcsec / 2:
                            return False

        return True

    @staticmethod
    def calculate_displacement_magnitude(
        displacements: Dict[str, Any],
        mirror_id: int
    ) -> Tuple[float, float]:
        """
        Вычисляет полную величину смещения для зеркала.

        Returns:
            (linear_magnitude_um, angular_magnitude_arcsec)
        """
        mirror_key = f'mirror_{mirror_id}'
        if mirror_key not in displacements:
            raise ValueError(f"Mirror {mirror_id} not found in displacements")

        mirror_data = displacements[mirror_key]

        # Линейные смещения
        linear_x = mirror_data['linear_x']['fine_value'] if 'fine_value' in mirror_data['linear_x'] else mirror_data['linear_x']['value']
        linear_y = mirror_data['linear_y']['fine_value'] if 'fine_value' in mirror_data['linear_y'] else mirror_data['linear_y']['value']
        linear_z = mirror_data['linear_z']['fine_value'] if 'fine_value' in mirror_data['linear_z'] else mirror_data['linear_z']['value']
        linear_mag = np.sqrt(linear_x**2 + linear_y**2 + linear_z**2)

        # Угловые смещения
        angular_x = mirror_data['angular_x']['fine_value'] if 'fine_value' in mirror_data['angular_x'] else mirror_data['angular_x']['value']
        angular_y = mirror_data['angular_y']['fine_value'] if 'fine_value' in mirror_data['angular_y'] else mirror_data['angular_y']['value']
        angular_mag = np.sqrt(angular_x**2 + angular_y**2)

        return linear_mag, angular_mag


# Удобные функции для создания декодера с разными настройками

def create_high_precision_decoder(
    bits_per_param: int = 8,
    use_hybrid: bool = True
) -> KorschDisplacementDecoder:
    """
    Создает декодер для сверхвысокой точности (<1 мкм).
    """
    return KorschDisplacementDecoder(
        bits_per_param=bits_per_param,
        linear_range_um=1000.0,  # ±1 мм
        angular_range_arcsec=60.0,  # ±1'
        use_hybrid=use_hybrid,
        regression_scale_factor=4.0
    )


def create_standard_decoder(
    bits_per_param: int = 6,
    use_hybrid: bool = False
) -> KorschDisplacementDecoder:
    """
    Создает стандартный декодер.
    """
    return KorschDisplacementDecoder(
        bits_per_param=bits_per_param,
        linear_range_um=1000.0,
        angular_range_arcsec=60.0,
        use_hybrid=use_hybrid,
        regression_scale_factor=2.0
    )


def estimate_required_bits(target_precision_um: float, target_precision_arcsec: float) -> Dict[str, int]:
    """
    Оценивает необходимое количество бит для достижения заданной точности.
    """
    linear_range_um = 1000.0  # ±1 мм
    angular_range_arcsec = 60.0  # ±1'

    required_linear_levels = linear_range_um / target_precision_um
    required_angular_levels = angular_range_arcsec / target_precision_arcsec

    required_linear_bits = np.ceil(np.log2(required_linear_levels))
    required_angular_bits = np.ceil(np.log2(required_angular_levels))

    return {
        'linear_bits': int(required_linear_bits),
        'angular_bits': int(required_angular_bits),
        'recommended_bits': max(int(required_linear_bits), int(required_angular_bits))
    }


if __name__ == "__main__":
    # Тестирование декодера
    decoder = create_high_precision_decoder(bits_per_param=7, use_hybrid=True)

    # Пример бинарного вектора
    test_bits = np.random.randint(0, 2, 70)  # 2 зеркала × 5 параметров × 7 бит
    test_regression = np.random.randn(10) * 0.1  # 10 регрессионных значений

    print("Precision stats:", decoder.get_precision_stats())

    # Декодирование
    displacements = decoder.decode_hybrid_output(test_bits, test_regression)

    print("\nDecoded displacements:")
    for mirror_name, mirror_data in displacements.items():
        print(f"\n{mirror_name}:")
        for param_name, param_info in mirror_data.items():
            print(f"  {param_name}: {param_info['fine_value']:.3f} {param_info['unit']}")

        # Величина смещения
        lin_mag, ang_mag = KorschDisplacementDecoder.calculate_displacement_magnitude(
            displacements, int(mirror_name.split('_')[1])
        )
        print(f"  Linear magnitude: {lin_mag:.3f} um")
        print(f"  Angular magnitude: {ang_mag:.3f} arcsec")