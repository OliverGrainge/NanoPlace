from pytorch_metric_learning import losses, miners


def load_miner(miner_name: str):
    if miner_name == "multisimilarity":
        return miners.MultiSimilarityMiner(epsilon=0.1)
    else:
        raise ValueError(f"Miner {miner_name} not found")


def load_loss(loss_name: str):
    if loss_name == "multisimilarity":
        return losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5)
    else:
        raise ValueError(f"Loss {loss_name} not found")
