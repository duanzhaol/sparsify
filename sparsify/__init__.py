__version__ = "1.3.0"

# Add project root to path so lowrank_encoder can be imported
import sys
from pathlib import Path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from .config import SaeConfig, SparseCoderConfig, TrainConfig
from .sparse_coder import Sae, SparseCoder
from .trainer import SaeTrainer, Trainer

__all__ = [
    "Sae",
    "SaeConfig",
    "SaeTrainer",
    "SparseCoder",
    "SparseCoderConfig",
    "Trainer",
    "TrainConfig",
]
