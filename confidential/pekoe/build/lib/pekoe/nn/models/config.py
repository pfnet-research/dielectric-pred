import hashlib
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import dacite
import yaml


@dataclass
class BaseConfig:
    version: str
    parent: pathlib.Path
    arch: str
    elements_supported: List[int]
    calc_modes: List[str]
    checksum: str
    default_calc_mode: str = "crystal"
    weights: Optional[pathlib.Path] = None
    parameters: Dict[Any, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(class_, path: pathlib.Path) -> "BaseConfig":
        with open(path) as f:
            content = f.read()
            d = yaml.safe_load(content)
            d["checksum"] = hashlib.md5(content.encode()).hexdigest()

        d["parent"] = path.parent
        return dacite.from_dict(
            data_class=class_,
            data=d,
            config=dacite.Config(cast=[pathlib.Path]),
        )

    @property
    def weights_path(self) -> Optional[pathlib.Path]:
        if self.weights is None:
            return None
        return self.parent / self.weights
