import argparse
import os

import numpy as np
import torch

from analyse import analyse_cossim
from datasets.val import get_val_dataset
from eval.utils import compute_descriptors
from models import get_model


def _desc_dtype(desc_dtype: str):
    if desc_dtype == "float16":
        return np.float16
    elif desc_dtype == "float32":
        return np.float32
    else:
        raise ValueError(f"Descriptor dtype {desc_dtype} not supported")


def main(args):
    model = get_model(args.model_name, pretrained=True)
    dataset = get_val_dataset(args.dataset_name, transform=model.transform)
    descriptors = compute_descriptors(
        model,
        dataset,
        desc_dtype=_desc_dtype(args.desc_dtype),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    query_descriptors = descriptors[: dataset.queries_num]
    database_descriptors = descriptors[dataset.queries_num :]
    ground_truth = dataset.ground_truth
    save_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"logs/cossim/{args.model_name}/{args.dataset_name}.png",
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    analyse_cossim(
        query_descriptors, database_descriptors, ground_truth, save_path=save_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="sf_xl_small")
    parser.add_argument("--model_name", type=str, default="cosplace")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--desc_dtype", type=str, default="float16")
    args = parser.parse_args()
    main(args)
