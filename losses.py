#3 losses.py
import torch
import torch.nn.functional as F

def multi_task_loss(outputs, batch, alpha=10.0, beta=1.0, lambda_seg =1.0, lambda_cls=0.1):
    """
    outputs: dict trả về từ model
    batch: dict từ DataLoader (datasets/casia.py)
      - batch["mask"]:  B x 1 x H x W
      - batch["label"]:B            (0/1)
      - batch["has_mask"]:B         (bool)

    alpha, beta: trọng số cho BCE pixel (tampered / pristine)
    lambda_seg, lambda_clf: trọng số cho 2 task
    """
    mask_logits = outputs["mask_logits"] # Bx1x 512 x 512
    det_logits = outputs["det_logits"] # B 
    device = mask_logits.device

    masks = batch["mask"].to(device).float()                    
    labels = batch["label"].to(device).float()

    has_mask = batch["has_mask"].to(device)  #bool
    # detection loss (BCE), all imgs
    det_loss = F.binary_cross_entropy_with_logits(det_logits, labels)
    # segmentation loss (BCE), chỉ imgs có mask
    if has_mask.any():
        idx = has_mask.nonzero(as_tuple=True)[0]
        m_logits = mask_logits[idx]  # Bx1x512x512
        m_gt = masks[idx]              # Bx1x512x512
        p = torch.sigmoid(m_logits)
        eps = 1e-6
        seg_loss = -(
            alpha * m_gt * torch.log(p + eps) +
            beta * (1 - m_gt) * torch.log(1 - p + eps)
        ).mean()
    else:
        seg_loss = torch.tensor(0.0, device=device)

    total_loss = lambda_seg * seg_loss + lambda_cls * det_loss
    return total_loss, seg_loss, det_loss
