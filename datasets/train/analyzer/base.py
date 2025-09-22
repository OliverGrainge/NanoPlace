from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd


class AnalyzerStep(ABC):
    """Abstract base class for curation steps"""

    @abstractmethod
    def __call__(
        self,
        dataconfig: pd.DataFrame,
        descriptors: Optional[np.ndarray] = None,
        **kwargs
    ) -> Tuple[pd.DataFrame, Union[np.ndarray, None]]:
        """
        Apply analyzer step

        Returns:
            (filtered_config, filtered_descriptors, stats)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Step name for logging and identification"""
        pass
