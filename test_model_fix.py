#!/usr/bin/env python3
"""
Тестовый скрипт для проверки исправления ошибки TypeError в InterferoNetMultiLabel
"""
import sys
import os
sys.path.append('src')

import torch

def test_interferonet_model():
    """Проверяем, что модель создается с новыми параметрами"""
    try:
        from models.net import InterferoNetMultiLabel

        print("✅ Testing basic model creation...")
        model = InterferoNetMultiLabel(
            out_dim=70,  # 10 parameters * 7 bits per parameter
            base_channels=32,
            use_hybrid_head=False,
            bits_per_parameter=7,
            use_reference_input=False
        )
        print(f"✅ Basic model created successfully")
        print(f"   Model bits_per_parameter: {model.bits_per_parameter}")
        print(f"   Model use_reference_input: {model.use_reference_input}")

        print("\n✅ Testing model with reference input...")
        model_ref = InterferoNetMultiLabel(
            out_dim=70,
            base_channels=32,
            use_hybrid_head=False,
            bits_per_parameter=7,
            use_reference_input=True
        )
        print(f"✅ Model with reference input created successfully")
        print(f"   Model bits_per_parameter: {model_ref.bits_per_parameter}")
        print(f"   Model use_reference_input: {model_ref.use_reference_input}")

        print("\n✅ Testing hybrid head model...")
        model_hybrid = InterferoNetMultiLabel(
            out_dim=70,
            base_channels=32,
            use_hybrid_head=True,
            regression_dim=10,
            bits_per_parameter=7,
            use_reference_input=False
        )
        print(f"✅ Hybrid model created successfully")
        print(f"   Model use_hybrid_head: {model_hybrid.use_hybrid_head}")
        print(f"   Model regression_dim: {model_hybrid.regression_dim}")

        # Тестирование forward pass
        print("\n✅ Testing forward pass...")
        batch_size = 2
        height, width = 224, 224

        # Тест с обычным входом (1 канал)
        x_single = torch.randn(batch_size, 1, height, width)
        output_single = model(x_single)
        print(f"✅ Single channel forward pass successful")
        print(f"   Input shape: {x_single.shape}")
        print(f"   Output shape: {output_single.shape}")

        # Тест с двойным входом (2 канала)
        x_dual = torch.randn(batch_size, 2, height, width)
        output_dual = model_ref(x_dual)
        print(f"✅ Dual channel forward pass successful")
        print(f"   Input shape: {x_dual.shape}")
        print(f"   Output shape: {output_dual.shape}")

        # Тест с гибридной головой
        x_hybrid = torch.randn(batch_size, 1, height, width)
        output_hybrid = model_hybrid(x_hybrid)
        if isinstance(output_hybrid, tuple):
            classification, regression = output_hybrid
            print(f"✅ Hybrid forward pass successful")
            print(f"   Classification shape: {classification.shape}")
            print(f"   Regression shape: {regression.shape}")
        else:
            print(f"❌ Hybrid model should return tuple, got {type(output_hybrid)}")
            return False

        print("\n✅ All tests passed! Model fixes work correctly.")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_train_compatibility():
    """Проверяем совместимость с train.py"""
    try:
        from models.net import InterferoNetMultiLabel

        print("\n✅ Testing train.py compatibility...")

        # Параметры как в train.py
        K = 70  # Например, 10 parameters * 7 bits
        bits_per_number = 7
        use_reference = False
        korsch_mode = True

        model_cfg = {
            "base_channels": 32,
            "width_multipliers": [2, 4, 8, 12],
            "block_repeats": [1, 1, 1, 1],
            "norm": "groupnorm",
            "gn_groups": 16,
            "stochastic_depth": 0.0,
            "dropout": 0.2,
            "head_dropout": 0.3,
            "head_norm": "layernorm",
            "use_hybrid_head": korsch_mode,
            "regression_dim": 10 if korsch_mode else None,
        }

        # Создание модели как в train.py:790-807
        model = InterferoNetMultiLabel(
            out_dim=K,
            base_channels=int(model_cfg.get("base_channels", 32)),
            width_multipliers=model_cfg.get("width_multipliers", [2, 4, 8, 12]),
            block_repeats=model_cfg.get("block_repeats", [1, 1, 1, 1]),
            norm_type=model_cfg.get("norm", "groupnorm"),
            gn_groups=int(model_cfg.get("gn_groups", 16)),
            stochastic_depth=float(model_cfg.get("stochastic_depth", 0.0)),
            dropout=float(model_cfg.get("dropout", 0.2)),
            hidden_dims=model_cfg.get("hidden_dims", None),
            head_dropout=float(model_cfg.get("head_dropout", 0.3)),
            head_norm=model_cfg.get("head_norm", "layernorm"),
            # Korsch and reference support
            bits_per_parameter=bits_per_number,
            use_hybrid_head=bool(model_cfg.get("use_hybrid_head", korsch_mode)),
            regression_dim=int(model_cfg.get("regression_dim", 10)) if korsch_mode else None,
            use_reference_input=use_reference,
        )

        print(f"✅ Model creation with train.py parameters successful!")
        print(f"   Model parameters match train.py expectations")

        return True

    except Exception as e:
        print(f"❌ Error in train.py compatibility test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success1 = test_interferonet_model()
    success2 = test_train_compatibility()

    if success1 and success2:
        print("\n🎉 All tests passed! The TypeError fix is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed.")
        sys.exit(1)