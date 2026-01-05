import torch
import torch.nn.functional as F

def dice_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float=1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits) #scale [0,1]
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)

    probs = probs.float()
    targets = targets.float()

    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    intersection = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1) 
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()

def multi_task_loss(
        outputs, batch,
        alpha: float = 10.0, beta:float = 1.0,
        lambda_seg: float = 1.0, lambda_cls:float=0.1, dice_weight:float=1.0,

):
    mask_logits = outputs["mask_logits"] #b,1,h,w
    det_logits = outputs["det_logits"]
    device = mask_logits.device
    masks = batch["mask"].to(device).float()  # Bx1xHxW
    labels = batch["label"].to(device).float().view(-1)
    det_logits = det_logits.view(-1)

    has_mask = batch["has_mask"].to(device)

    det_loss = F.binary_cross_entropy_with_logits(det_logits, labels)

    if has_mask.any():
        idx = has_mask.nonzero(as_tuple=True)[0]
        m_logits = mask_logits[idx]
        m_gt = masks[idx]

        pos_weight = torch.tensor([alpha / max(beta, 1e-12)], device=device, dtype=m_logits.dtype)
        bce = F.binary_cross_entropy_with_logits(m_logits, m_gt, pos_weight=pos_weight)

        dloss = dice_loss_with_logits(m_logits, m_gt)

        seg_loss = bce + dice_weight * dloss
    else:
        seg_loss = torch.tensor(0.0, device=device)

    total_loss = lambda_seg * seg_loss + lambda_cls * det_loss
    return total_loss, seg_loss, det_loss