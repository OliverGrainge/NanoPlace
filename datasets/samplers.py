import torch
import numpy as np
from torch.utils.data import Sampler
from collections import defaultdict
import random


class MPerClassSampler(Sampler):
    """Sampler that ensures each batch contains m samples per class.

    This is essential for metric learning as it ensures positive pairs
    are available in each batch for effective mining and loss computation.
    """

    def __init__(self, labels, m, batch_size=None, length_before_new_iter=100000):
        """
        Args:
            labels: List of labels for each sample
            m: Number of samples per class in each batch
            batch_size: Total batch size (if None, will be computed)
            length_before_new_iter: Number of samples to yield before starting new iteration
        """
        self.labels = np.array(labels)
        self.m = m
        self.length_before_new_iter = length_before_new_iter

        # Group indices by class
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.class_to_indices[label].append(idx)

        self.classes = list(self.class_to_indices.keys())
        self.num_classes = len(self.classes)

        # Compute batch size if not provided
        if batch_size is None:
            self.batch_size = self.num_classes * self.m
        else:
            self.batch_size = batch_size

        # Ensure batch size is compatible
        if self.batch_size % self.m != 0:
            raise ValueError(
                f"Batch size {self.batch_size} must be divisible by m={self.m}"
            )

        self.classes_per_batch = self.batch_size // self.m

        # Filter classes that have at least m samples
        self.valid_classes = [
            cls for cls in self.classes if len(self.class_to_indices[cls]) >= self.m
        ]

        if len(self.valid_classes) == 0:
            raise ValueError("No classes have at least m samples")

        print(
            f"MPerClassSampler: {len(self.valid_classes)} valid classes, "
            f"{self.classes_per_batch} classes per batch, {self.m} samples per class"
        )

    def __iter__(self):
        for _ in range(self.length_before_new_iter):
            # Randomly select classes for this batch
            selected_classes = random.sample(
                self.valid_classes, min(self.classes_per_batch, len(self.valid_classes))
            )

            # Sample m indices from each selected class
            batch_indices = []
            for cls in selected_classes:
                class_indices = self.class_to_indices[cls]
                # Sample with replacement if class has fewer than m samples
                if len(class_indices) >= self.m:
                    sampled = random.sample(class_indices, self.m)
                else:
                    sampled = random.choices(class_indices, k=self.m)
                batch_indices.extend(sampled)

            # Shuffle the batch
            random.shuffle(batch_indices)

            for idx in batch_indices:
                yield idx

    def __len__(self):
        return self.length_before_new_iter


class BalancedBatchSampler(Sampler):
    """Alternative sampler that ensures balanced representation of classes."""

    def __init__(self, labels, batch_size, num_classes_per_batch=None):
        """
        Args:
            labels: List of labels for each sample
            batch_size: Total batch size
            num_classes_per_batch: Number of different classes per batch
        """
        self.labels = np.array(labels)
        self.batch_size = batch_size

        # Group indices by class
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.class_to_indices[label].append(idx)

        self.classes = list(self.class_to_indices.keys())
        self.num_classes = len(self.classes)

        if num_classes_per_batch is None:
            self.num_classes_per_batch = min(self.num_classes, batch_size)
        else:
            self.num_classes_per_batch = num_classes_per_batch

        self.samples_per_class = self.batch_size // self.num_classes_per_batch

        print(
            f"BalancedBatchSampler: {self.num_classes} classes, "
            f"{self.num_classes_per_batch} classes per batch, "
            f"{self.samples_per_class} samples per class"
        )

    def __iter__(self):
        # Create batches
        all_indices = list(range(len(self.labels)))
        random.shuffle(all_indices)

        for i in range(0, len(all_indices), self.batch_size):
            batch_indices = all_indices[i : i + self.batch_size]
            if len(batch_indices) == self.batch_size:
                yield batch_indices

    def __len__(self):
        return len(self.labels) // self.batch_size




