# TODO merge naive and weighted loss.
import math
import torch
import torch.nn.functional as F
from ..bbox import delta2bbox

def weighted_nll_loss(pred, label, weight, avg_factor=None):
    if avg_factor is None:
        avg_factor = max(torch.sum(weight > 0).float().item(), 1.)
    raw = F.nll_loss(pred, label, reduction='none')
    return torch.sum(raw * weight)[None] / avg_factor


def weighted_cross_entropy(pred, label, weight, avg_factor=None, reduce=True):
    if avg_factor is None:
        avg_factor = max(torch.sum(weight > 0).float().item(), 1.)
    raw = F.cross_entropy(pred, label, reduction='none')
    if reduce:
        return torch.sum(raw * weight)[None] / avg_factor
    else:
        return raw * weight / avg_factor


def weighted_binary_cross_entropy(pred, label, weight, avg_factor=None):
    if avg_factor is None:
        avg_factor = max(torch.sum(weight > 0).float().item(), 1.)
    return F.binary_cross_entropy_with_logits(
        pred, label.float(), weight.float(),
        reduction='sum')[None] / avg_factor


def sigmoid_focal_loss(pred,
                       target,
                       weight,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean'):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    weight = (alpha * target + (1 - alpha) * (1 - target)) * weight
    weight = weight * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * weight
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weighted_sigmoid_focal_loss(pred,
                                target,
                                weight,
                                gamma=2.0,
                                alpha=0.25,
                                avg_factor=None,
                                num_classes=80):
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / num_classes + 1e-6
    return sigmoid_focal_loss(
        pred, target, weight, gamma=gamma, alpha=alpha,
        reduction='sum')[None] / avg_factor


def mask_cross_entropy(pred, target, label):
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, reduction='mean')[None]


def smooth_l1_loss(pred, target, beta=1.0, reduction='mean'):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.sum() / pred.numel()
    elif reduction_enum == 2:
        return loss.sum()


def weighted_smoothl1(pred, target, weight, beta=1.0, avg_factor=None):
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / 4 + 1e-6
    loss = smooth_l1_loss(pred, target, beta, reduction='none')
    return torch.sum(loss * weight)[None] / avg_factor

def generalized_iou_loss(pred, target, anchors, means, stds, reduction='mean'):
    assert pred.size() == target.size() and target.numel() > 0

    bboxes1 = delta2bbox(anchors, pred, means, stds)
    bboxes2 = delta2bbox(anchors, target, means, stds)
    lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
    rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

    wh = (rb - lt + 1).clamp(min=0)  # [rows, 2]
    overlap = wh[:, 0] * wh[:, 1]

    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
        bboxes1[:, 3] - bboxes1[:, 1] + 1)

    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
        bboxes2[:, 3] - bboxes2[:, 1] + 1)
    ious = overlap / (area1 + area2 - overlap + 1e-7)

    lt = torch.min(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
    rb = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]
    wh = (rb - lt + 1).clamp(min=0)  # [rows, 2]
    ac = wh[:, 0] * wh[:, 1]
    loss = 1 - ious + (ac - overlap) / (ac + 1e-7) #1 - (ious - (ac - overlap) / ac)

    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.sum() / pred.numel()
    elif reduction_enum == 2:
        return loss.sum()

def weighted_generalized_iou(pred, target, anchors, means, stds, weight, avg_factor=None):
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / 4 + 1e-6
    loss = generalized_iou_loss(pred, target, anchors, means, stds, reduction='none')
    return torch.sum(loss * weight)[None] / avg_factor

def accuracy(pred, target, topk=1):
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    _, pred_label = pred.topk(maxk, 1, True, True)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / pred.size(0)))
    return res[0] if return_single else res

class MultilossAdaptiveBalancer(torch.nn.Module):
    """Implements multiple losses balancing approach from https://arxiv.org/pdf/1705.07115.pdf"""
    def __init__(self, num_losses):
        super(MultilossAdaptiveBalancer, self).__init__()
        self.weight = torch.nn.Parameter(torch.FloatTensor(num_losses))
        for i in range(num_losses):
            self.weight.data[i] = 0.

    def get_weights(self):
        return [math.exp(-s.item()) for s in self.weight.data]

    def forward(self, losses):
        assert isinstance(losses, list)
        assert len(losses) == self.weight.shape[0]

        total_loss = 0
        for i, loss in enumerate(losses):
            total_loss += torch.exp(-self.weight[i])*loss + 0.5*self.weight[i]

        return total_loss

class MultilossSummator(torch.nn.Module):
    def __init__(self, num_losses):
        super(MultilossSummator, self).__init__()
        self.num_losses = num_losses

    def get_weights(self):
        return [1.]*self.num_losses

    def forward(self, losses):
        assert isinstance(losses, list)
        return sum(losses)
