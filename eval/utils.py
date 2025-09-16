from typing import Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import faiss
from datasets.val.base import ValDataset

def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalize with numerical safety."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    np.maximum(norms, eps, out=norms)
    return x / norms


def _as_numpy(x: torch.Tensor, dtype: np.dtype) -> np.ndarray:
    """Move tensor to CPU and convert to desired numpy dtype."""
    return x.detach().cpu().numpy().astype(dtype, copy=False)


def _first_output(x):
    """Return the first element if model returns tuple/list, else the tensor itself."""
    return x[0] if isinstance(x, (tuple, list)) else x


def _faiss_ip_index(dim: int, use_gpu: bool = False):
    """Build an inner-product FAISS index (optionally on GPU)."""
    index = faiss.IndexFlatIP(dim)
    if use_gpu:
        # Requires faiss-gpu installed; otherwise this will raise.
        index = faiss.index_cpu_to_all_gpus(index)
    return index


def compute_descriptors(
    model: nn.Module,
    dataset: ValDataset,
    desc_dtype: np.dtype,
    batch_size: int = 32,
    num_workers: int = 4,
    pbar: bool = True,
) -> np.ndarray:
    """
    Compute L2-normalized descriptors for all images in `dataset`.

    Args:
        model: Torch module producing descriptors (N, D) or (tensor, ...).
        dataset: Dataset yielding (image_tensor, index).
        desc_dtype: Numpy dtype for the output descriptors (e.g., np.float16/32).
        batch_size: DataLoader batch size.
        num_workers: DataLoader workers.
        pbar: Show progress bar.

    Returns:
        np.ndarray of shape (len(dataset), descriptor_dim), L2-normalized.
    """
    n = dataset.images_num if hasattr(dataset, "images_num") else len(dataset)
    descriptors = np.zeros((n, model.descriptor_dim), dtype=desc_dtype)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    # inference_mode gives a bit more speed than no_grad and enforces no autograd state.
    with torch.inference_mode():
        iterator = tqdm(dataloader, disable=not pbar, total=len(dataloader))
        for images, idx in iterator:
            images = images.to(device, non_blocking=True)

            out = _first_output(model(images))
            if out.dim() > 2:
                # Flatten everything but batch if the model returns (B, D, ...)
                out = out.view(out.size(0), -1)

            if out.size(1) != model.descriptor_dim:
                raise ValueError(
                    f"Descriptor dim mismatch: got {out.size(1)}, expected {model.descriptor_dim}"
                )

            batch_desc = _as_numpy(out, desc_dtype)
            descriptors[idx.numpy()] = batch_desc

    # Row-wise L2 normalization (safe)
    descriptors = _l2_normalize_rows(descriptors)
    return descriptors


def match_cosine(
    query_descriptors: np.ndarray,
    database_descriptors: np.ndarray,
    k: int = 1,
    use_gpu: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cosine matching via FAISS inner product. Assumes inputs are (approximately) L2-normalized.

    FAISS expects float32; we convert as needed. If your inputs are not normalized,
    cosine similarity will be wrong—normalize first.

    Args:
        query_descriptors: (Nq, D) L2-normalized query descriptors.
        database_descriptors: (Nd, D) L2-normalized database descriptors.
        k: top-k to retrieve.
        use_gpu: use FAISS GPU if available and faiss-gpu is installed.

    Returns:
        indices: (Nq, k) int32 indices into database rows.
        scores:  (Nq, k) float32 cosine similarities.
    """
    if query_descriptors.ndim != 2 or database_descriptors.ndim != 2:
        raise ValueError("Descriptors must be 2D arrays.")
    if query_descriptors.shape[1] != database_descriptors.shape[1]:
        raise ValueError("Query and database dims must match.")

    # Ensure float32 for FAISS and L2-normalize just in case.
    q = _l2_normalize_rows(np.asarray(query_descriptors, dtype=np.float32, order="C"))
    d = _l2_normalize_rows(np.asarray(database_descriptors, dtype=np.float32, order="C"))

    index = _faiss_ip_index(dim=q.shape[1], use_gpu=use_gpu)
    index.add(d)  # (Nd, D)
    scores, indices = index.search(q, k)  # (Nq, k)
    return indices, scores


def correct_matches(matches: np.ndarray, ground_truth: np.ndarray, k: int = 1) -> np.ndarray:
    """
    For each query i, return True if any of the top-k matches[i] appear in ground_truth[i].

    Args:
        matches: (N, Kmax) array of DB indices (e.g., FAISS result).
        ground_truth: (N, G_i) ragged-like object or (N, G) array of ground-truth indices.
        k: top-k to consider from `matches`.

    Returns:
        Boolean array of shape (N,) indicating correctness.
    """
    topk = matches[:, :k]
    # Ground truth may be ragged (list of arrays) or dense ndarray per row.
    # np.isin is row-wise here, so a tiny loop is still the cleanest/most correct.
    return np.array([np.any(np.isin(topk[i], ground_truth[i])) for i in range(len(topk))], dtype=bool)
