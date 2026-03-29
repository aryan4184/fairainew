import torch


class AverageMeter:
    """Tracks and computes running average of a metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        """
        Update meter with a new value.

        Args:
            val (float): metric value
            n (int): number of samples
        """
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count
        else:
            self.avg = 0.0

def accuracy(output, target, topk=(1,)):
    """
    Compute top-k accuracy.

    Args:
        output (torch.Tensor): model outputs (logits)
        target (torch.Tensor): ground truth labels
        topk (tuple): values of k for top-k accuracy

    Returns:
        list[torch.Tensor]: accuracies for each k
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res