import argparse
import os
import sys
from typing import List

import pandas as pd
import yaml

from datasets.train.analyzer import (
    ClassCounts,
    ClassSamples,
    InterUTMDist,
    IntraUTMDist,
    NanoPlaceAnalyzer,
)

from datasets.train.pipeline import (
    AggroClust,
    Embeddings,
    NanoPlaceDataPipeline,
    RandomClassSelection, 
    RandomInstanceSelection,
    GreedyIntraClassSelection,
    MinPerClass,
    ConfusableClassSelection,
)


def get_curation_step(name: str, **kwargs):
    name = name.lower()
    if name == "aggroclust":
        return AggroClust(**kwargs)
    elif name == "embeddings":
        return Embeddings(**kwargs)
    elif name == "randomclassselection":
        return RandomClassSelection(**kwargs)
    elif name == "randominstanceselection":
        return RandomInstanceSelection(**kwargs)
    elif name == "greedyintraclassselection":
        return GreedyIntraClassSelection(**kwargs)
    elif name == "minperclass":
        return MinPerClass(**kwargs)
    elif name == "confusableclassselection":
        return ConfusableClassSelection(**kwargs)
    else:
        raise ValueError(f"Curation step {name} not found")


def get_analyzer_step(name: str):
    name = name.lower()
    if name == "utmdist":
        return IntraUTMDist()
    elif name == "interutmdist":
        return InterUTMDist()
    elif name == "classsamples":
        return ClassSamples()
    elif name == "classcounts":
        return ClassCounts()
    else:
        raise ValueError(f"Analyzer step {name} not found")


def read_curationconfig():
    if len(sys.argv) > 1:
        curationconfig_path = sys.argv[1]
        with open(curationconfig_path, "r") as file:
            curationconfig = yaml.load(file, Loader=yaml.SafeLoader)
        return curationconfig
    else:
        print("No config file provided.")


def load_curation_steps(config: dict):
    curation_steps = []
    for step in config["curation_steps"]:
        curation_steps.append(get_curation_step(step, **config["curation_steps"][step]))
    return curation_steps


def load_analyzer_steps(config: dict):
    analyzer_steps = []
    for step in config["analyzer_steps"]:
        analyzer_steps.append(get_analyzer_step(step))
    return analyzer_steps


def load_save_dir(config: dict):
    return os.path.join(config["save_dir"], config["name"])


def load_image_dir(config: dict):
    return config["image_dir"]


def save_dataconfig(dataconfig: pd.DataFrame, config: dict):
    dataconfig.to_parquet(
        os.path.join(config["save_dir"], config["name"], "dataconfig.parquet")
    )


def main():
    config = read_curationconfig()
    image_dir = load_image_dir(config)
    curation_steps = load_curation_steps(config)
    analyzer_steps = load_analyzer_steps(config)

    pipeline = NanoPlaceDataPipeline(
        image_dir=image_dir,
        CurationSteps=curation_steps,
        save_dir=config["save_dir"],
        debug=config["debug"],
    )
    analyzer = NanoPlaceAnalyzer(
        analyzer_steps=analyzer_steps, save_dir=config["save_dir"]
    )

    print("🚀 Starting NanoPlace Curation")
    print(f"📁 Save directory: {config['save_dir']}")
    print(f"📂 Image directories: {len(image_dir)} directories")

    # Run pipeline
    dataconfig, descriptors = pipeline.run()

    # Run analyzer
    dataconfig, descriptors = analyzer.run(dataconfig, descriptors, plot=True)

    print("\n🎉 All processing complete!")


if __name__ == "__main__":
    main()
