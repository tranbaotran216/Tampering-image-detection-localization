# eval det  + loc
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def compute_pixels_stats(pred, gt):
    # out: tp, fp, tn, fn (int)
    pred = pred.bool()
    gt = gt.bool()
    tp = (pred & gt).sum().item()
    fp =(pred & ~gt).sum().item()
    tn = (~pred & ~gt).sum().item()
    fn = (~pred & gt).sum().item()
    return tp, fp, tn, fn

def calc_pixel_metrics(tp, fp, tn, fn, espsilon=1e-6):
    precision = tp / (tp + fp + espsilon)
    recall = tp / (tp + fn + espsilon)
    f1 = 2 * precision * recall / (precision + recall + espsilon)
    iou = tp / (tp + fp + fn + espsilon)

    # mcc 
    mcc_num = tp * tn - fp * fn
    mcc_den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + espsilon) ** 0.5
    mcc = mcc_num / mcc_den

    return {
        "pixel_precision": precision,
        "pixel_recall": recall,
        "pixel_f1": f1,
        "pixel_iou": iou,
        "pixel_mcc": mcc,
    }

@torch.no_grad()
def evaluate(model, dataloader, device="cuda", threshold=0.5):
    model.eval()
    all_labels = []
    all_probs = []

    tp = fp = tn = fn = 0  # pixel-level

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        imgs = batch["image"].to(device)          # Bx3xHxW
        masks = batch["mask"].to(device)        # Bx1xHxW
        labels = batch["label"].float().to(device)      # B
        has_mask = batch["has_mask"].to(device) # B

        outputs = model(imgs)
        det_prob = outputs["det_prob"] #B
        mask_prob = outputs["mask_prob"] # Bx1xHxW
        # Detection
        all_labels.append(labels.cpu().numpy())
        all_probs.append(det_prob.detach().cpu().numpy())
        
        # Localization (chỉ imgs có mask)
        if has_mask.any():
            idx = has_mask.nonzero(as_tuple=True)[0]
            m_prob = mask_prob[idx]  # Bx1xHxW
            m_gt = masks[idx]        # Bx1xHxW

            pred_mask = (m_prob > threshold).float()
            gt_mask = (m_gt > 0.5).float()
            tp_i, fp_i, tn_i, fn_i = compute_pixels_stats(pred_mask, gt_mask)
            tp += tp_i
            fp += fp_i
            tn += tn_i
            fn += fn_i
    # Detection metrics
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    pred_cls = (all_probs >= 0.5).astype(np.int64)
    acc = accuracy_score(all_labels, pred_cls)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, pred_cls, average="binary", zero_division=0
    )

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan") 

    det_metrics = {
        "det_acc": acc,
        "det_prec": prec,
        "det_rec": rec,
        "det_f1": f1,
        "det_auc": auc,
    }
    if tp + fp + fn > 0:
        loc_metrics = calc_pixel_metrics(tp, fp, tn, fn)
    else:
        loc_metrics = {
            "pixel_precision": float("nan"),
            "pixel_recall": float("nan"),
            "pixel_f1": float("nan"),
            "pixel_iou": float("nan"),
            "pixel_mcc": float("nan"),
        }

    det_metrics.update(loc_metrics)
    return det_metrics