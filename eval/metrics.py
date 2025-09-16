import numpy as np

def compute_recall(
    matches: np.ndarray, 
    ground_truth: np.ndarray, 
    k: int = 1
) -> float:
    """
    Compute recall@k for a set of predictions against ground truth.

    Args:
        matches (np.ndarray): Array of predicted matches, shape (N, M).
        ground_truth (np.ndarray): Array of ground truth labels, shape (N, G).
        k (int, optional): The number of top predictions to consider. Defaults to 1.

    Returns:
        float: Recall@k percentage (0–100).
    """
    top_k_matches = matches[:, :k]
    num_correct = sum(
        np.any(np.isin(top_k_matches[i], ground_truth[i]))
        for i in range(len(top_k_matches))
    )
    return (num_correct / len(top_k_matches)) * 100