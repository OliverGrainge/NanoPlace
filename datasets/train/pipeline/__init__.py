import math
import os
from abc import ABC, abstractmethod
from re import L
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pyproj import Transformer

from datasets.train.pipeline.aggroclust import AggroClust
from datasets.train.pipeline.base import CurationStep
from datasets.train.pipeline.embeddings import Embeddings
from datasets.train.pipeline.greedyintraclassselection import GreedyIntraClassSelection
from datasets.train.pipeline.randomclassselection import RandomClassSelection
from datasets.train.pipeline.randominstanceselection import RandomInstanceSelection
from datasets.train.utils import read_gmberton_utm, read_images
from datasets.train.pipeline.minperclass import MinPerClass
from datasets.train.pipeline.confusableclassselection import ConfusableClassSelection


def lat_lon_to_utm(row):
    lat, lon = row["lat"], row["lon"]
    utm_zone = int(math.floor((lon + 180) / 6) + 1)
    hemisphere = "N" if lat >= 0 else "S"
    utm_crs = f"+proj=utm +zone={utm_zone} +{hemisphere.lower()} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    easting, northing = transformer.transform(lon, lat)
    return easting, northing


class NanoPlaceDataPipeline(ABC):
    def __init__(
        self,
        image_dir: str,
        CurationSteps: List[CurationStep],
        save_dir: str,
        debug: bool = False,
    ):
        self.image_dir = image_dir
        self.steps = CurationSteps
        self.save_dir = save_dir
        self.step_stats = []  # Track stats for each step
        self.debug = debug

    def _save_config(self, dataconfig: pd.DataFrame):
        dataconfig.to_pickle(os.path.join(self.save_dir, "dataconfig.pkl"))

    def _print_stats(
        self,
        step_name: str,
        before_count: int,
        after_count: int,
        before_classes: int = None,
        after_classes: int = None,
    ):
        removed = before_count - after_count
        percentage = (removed / before_count) * 100 if before_count > 0 else 0

        # Store stats
        step_stat = {
            "step": step_name,
            "samples_before": before_count,
            "samples_after": after_count,
            "samples_removed": removed,
            "removal_percentage": percentage,
        }

        if before_classes is not None and after_classes is not None:
            classes_removed = before_classes - after_classes
            step_stat.update(
                {
                    "classes_before": before_classes,
                    "classes_after": after_classes,
                    "classes_removed": classes_removed,
                }
            )
            print(f"  ▶ {step_name}")
            print(f"    Samples: {before_count:,} → {after_count:,}")
            print(f"    Classes: {before_classes:,} → {after_classes:,}")
        else:
            print(f"  ▶ {step_name}: {before_count:,} → {after_count:,}")

        self.step_stats.append(step_stat)

    def print_summary(self):
        """Print a summary of all curation steps"""
        if not self.step_stats:
            return

        print("\n" + "=" * 80)
        print("📋 CURATION SUMMARY")
        print("=" * 80)

        for stat in self.step_stats:
            print(f"  {stat['step']}:")
            print(f"    Samples remaining: {stat['samples_after']:,}")
            if "classes_after" in stat:
                print(f"    Classes remaining: {stat['classes_after']:,}")

        total_final = self.step_stats[-1]["samples_after"] if self.step_stats else 0
        final_classes = (
            self.step_stats[-1].get("classes_after") if self.step_stats else None
        )

        print(f"\n  📊 FINAL DATASET:")
        print(f"    Samples: {total_final:,}")
        if final_classes is not None:
            print(f"    Classes: {final_classes:,}")
        print("=" * 80)

    def __getdataframes(self, csv_paths: List[str]):
        # Only load CSVs
        csv_paths = [p for p in csv_paths if p.lower().endswith(".csv")]
        assert (
            len(csv_paths) > 0
        ), f"No CSVs found in {os.path.join(self.image_dir, 'Dataframes')}"

        # Read first CSV
        df = pd.read_csv(csv_paths[0]).sample(frac=1)

        # Append others with city prefix to place_id
        for i in range(1, len(csv_paths)):
            tmp_df = pd.read_csv(csv_paths[i]).sample(frac=1)
            tmp_df["place_id"] = tmp_df["place_id"] + (i * 10**5)
            df = pd.concat([df, tmp_df], ignore_index=True)

        # Keep all rows (one per image). If you later need per-place indexing:
        # df = df.set_index('place_id')

        # Build filename
        df["filename"] = df.apply(self.get_img_name, axis=1)

        # IMPORTANT: Build absolute path: <image_dir>/Images/<city>/<filename>
        df["image_path"] = df.apply(
            lambda r: os.path.join(
                self.image_dir, "Images", str(r["city_id"]), r["filename"]
            ),
            axis=1,
        )

        # For downstream steps
        df["class_id"] = df["place_id"]
        return df

    def get_img_name(self, row):
        # Build the filename only (not the full path)
        city = row["city_id"]

        # Use the true place_id from the row
        pl_id = int(row["place_id"]) % 10**5
        pl_id = str(pl_id).zfill(7)

        panoid = row["panoid"]
        year = str(row["year"]).zfill(4)
        month = str(row["month"]).zfill(2)
        northdeg = str(row["northdeg"]).zfill(3)
        lat, lon = str(row["lat"]), str(row["lon"])

        return f"{city}_{pl_id}_{year}_{month}_{northdeg}_{lat}_{lon}_{panoid}.jpg"

    def setup_dataconfig(self) -> pd.DataFrame:
        csv_dir = os.path.join(self.image_dir, "Dataframes")
        csv_paths = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir)]
        csv_paths.sort()
        csv_paths = csv_paths[:1]
        dataconfig = self.__getdataframes(csv_paths)
        # dataconfig['easting'] = dataconfig.apply(lambda row: lat_lon_to_utm(row)[0], axis=1)
        # dataconfig['northing'] = dataconfig.apply(lambda row: lat_lon_to_utm(row)[1], axis=1)
        if self.debug:
            class_ids = dataconfig["class_id"].unique()
            class_ids = np.random.choice(class_ids, size=100, replace=False)
            dataconfig = dataconfig[dataconfig["class_id"].isin(class_ids)]
        # Assign a unique image_id column from 0 to len(dataconfig)-1 (not as index)
        return dataconfig

    def run(self) -> Tuple[pd.DataFrame, np.ndarray]:
        print("\n" + "=" * 80)
        print("🔧 DATA CURATION PIPELINE")
        print("=" * 80)

        dataconfig = self.setup_dataconfig()
        descriptors = None

        print(f"📊 Starting with {len(dataconfig)} samples")

        for step in self.steps:
            before_count = len(dataconfig)
            before_classes = (
                dataconfig["class_id"].nunique()
                if "class_id" in dataconfig.columns
                else None
            )

            dataconfig, descriptors = step(dataconfig, descriptors)

            after_count = len(dataconfig)
            after_classes = (
                dataconfig["class_id"].nunique()
                if "class_id" in dataconfig.columns
                else None
            )

            step_name = (
                step.name() if callable(getattr(step, "name", None)) else str(step)
            )
            self._print_stats(
                step_name, before_count, after_count, before_classes, after_classes
            )

        print("=" * 80)
        print(
            f"✅ Pipeline complete: {len(dataconfig)} samples, {dataconfig['class_id'].nunique() if 'class_id' in dataconfig.columns else 'N/A'} classes"
        )
        self.print_summary()  # Add this at the end
        self._save_config(dataconfig)
        return dataconfig, descriptors


    def name(self) -> str:
        desc = "NanoPlaceDataPipeline(\n  steps=["
        for step in self.steps:
            step_name = (
                step.name() if callable(getattr(step, "name", None)) else str(step)
            )
            desc += f"\n    {step_name},"
        desc += "\n  ]\n)"
        return desc
