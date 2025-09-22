import os
from typing import List, Optional

import numpy as np
import pandas as pd

from .base import AnalyzerStep
from .classcounts import ClassCounts
from .classsamples import ClassSamples
from .interutmdist import InterUTMDist
from .intrautmdist import IntraUTMDist
from .plotter import Plotter


class NanoPlaceAnalyzer:
    def __init__(self, analyzer_steps: List[AnalyzerStep], save_dir: str):
        self.steps = analyzer_steps
        self.save_dir = save_dir
        self.plotter = Plotter(save_dir)

    def _save_config(self, dataconfig: pd.DataFrame):
        dataconfig.to_pickle(os.path.join(self.save_dir, "dataconfig.pkl"))

    def run(
        self,
        dataconfig: pd.DataFrame,
        descriptors: Optional[np.ndarray] = None,
        plot: bool = True,
    ):
        print("\n" + "=" * 80)
        print("📈 ANALYSIS PIPELINE")
        print("=" * 80)

        print(
            f"📊 Analyzing {len(dataconfig)} samples, {dataconfig['class_id'].nunique() if 'class_id' in dataconfig.columns else 'N/A'} classes"
        )

        dataconfig.attrs["stats"] = {}
        for step in self.steps:
            print(f"  ▶ Running {step.name()}...")
            dataconfig, descriptors = step(dataconfig, descriptors)

        print("=" * 80)
        print("✅ Analysis complete")

        if plot:
            print("📊 Generating plots...")
            self.plotter.add_stats(dataconfig.attrs["stats"])
            self.plotter.plot()

        self._save_config(dataconfig)
        return dataconfig, descriptors

    def name(self) -> str:
        """
        Returns a human-readable description of the analyzer and its steps.
        """
        desc = "NanoPlaceAnalyzer(\n  steps=["
        for step in self.steps:
            step_name = step.name()
            desc += f"\n    {step_name},"
        desc += "\n  ]\n)"
        return desc
