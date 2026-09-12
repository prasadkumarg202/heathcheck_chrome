"""
Unified Configuration System for AuraPulse rPPG Platform.
Manages hyper-parameters for preprocessing, classical algorithms (POS, CHROM, ICA, GREEN, LGI, PBV),
deep learning models (PhysNet, TS-CAN, EfficientPhys, DeepPhys, BigSmall), datasets, and edge inference.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import os
import json


@dataclass
class PreprocessConfig:
    do_crop_face: bool = True
    use_large_face_box: bool = True
    large_box_coef: float = 1.5
    do_dynamic_detection: bool = False
    dynamic_detection_frequency: int = 30
    resize_w: int = 128
    resize_h: int = 128
    chunk_length: int = 180
    data_aug: List[str] = field(default_factory=lambda: ["None"])


@dataclass
class DatasetConfig:
    name: str = "UBFC-rPPG"
    data_path: str = "data/UBFC-rPPG"
    fs: float = 30.0
    cached_path: str = "PreprocessedData"
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)


@dataclass
class ModelConfig:
    name: str = "TSCAN"  # Physnet, Tscan, EfficientPhys, DeepPhys, BigSmall
    model_dir: str = "PreTrainedModels"
    drop_rate: float = 0.0
    frame_num: int = 64
    frame_depth: int = 10


@dataclass
class InferenceConfig:
    batch_size: int = 4
    evaluation_method: str = "FFT"  # FFT, Peak, Welch
    window_size: float = 10.0
    min_measurement_duration_s: float = 6.0
    hr_low_hz: float = 0.7
    hr_high_hz: float = 3.5
    min_sqi_threshold: float = 35.0


@dataclass
class UnsupervisedConfig:
    methods: List[str] = field(default_factory=lambda: ["POS", "CHROM", "PBV", "LGI", "GREEN", "ICA"])
    metrics: List[str] = field(default_factory=lambda: ["MAE", "RMSE", "MAPE", "Pearson", "SNR"])


@dataclass
class AuraPulseConfig:
    toolbox_mode: str = "unsupervised_method"  # train_and_test, only_test, unsupervised_method
    device: str = "cuda:0"
    train_dataset: Optional[DatasetConfig] = None
    valid_dataset: Optional[DatasetConfig] = None
    test_dataset: Optional[DatasetConfig] = None
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    unsupervised: UnsupervisedConfig = field(default_factory=UnsupervisedConfig)

    @classmethod
    def load_config(cls, file_path: str) -> AuraPulseConfig:
        if file_path.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            try:
                import yaml
                with open(file_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
            except ImportError:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)
