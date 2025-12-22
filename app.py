import os
import io

import numpy as np
from PIL import Image

import streamlit as st

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from models.mobilenet import MobileNetV2_SRM_DetLoc


# ======================
# Config Streamlit page
# ======================
st.set_page_config(
    page_title="Splicing Detection (MobileNetV2+SRM)",
    layout="wide",
)


# ======================
# Helper
# ======================
def build_transform(img_size: int):
    """Resize + ToTensor giống pipeline train."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),          # [0,1], 3xHxW
    ])


@st.cache_resource(show_spinner=False)
def load_model(ckpt_path: str, device_str: str):
    """Load model + checkpoint, cache cho session."""
    device = torch.device(device_str)
    model = MobileNetV2_SRM_DetLoc().to(device)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint không tồn tại: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    # Có thể là dict {"model": state_dict} hoặc state_dict thuần
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.eval()
    return model, device


def make_overlay(original: Image.Image, mask_np: np.ndarray, alpha: float = 0.5):
    """
    Tạo ảnh overlay: vùng mask (0..1) tô đỏ lên ảnh gốc với độ trong suốt alpha.
    original: PIL RGB
    mask_np: HxW (0..1)
    """
    img_np = np.array(original).astype(np.float32)  # HxWx3
    if img_np.max() <= 1.0:
        img_np *= 255.0

    img_np = img_np / 255.0  # 0..1

    # chuẩn hóa mask 0..1
    m = mask_np.astype(np.float32)
    if m.max() > 1.0:
        m = m / 255.0
    m = np.clip(m, 0.0, 1.0)

    # tạo overlay đỏ
    H, W = m.shape
    color = np.zeros((H, W, 3), dtype=np.float32)
    color[..., 0] = 1.0  # kênh R = 1, G=B=0

    m_exp = m[..., None]  # HxWx1
    overlay = img_np * (1 - alpha * m_exp) + color * (alpha * m_exp)
    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
    return overlay


# ======================
# Sidebar
# ======================
st.sidebar.header("Cấu hình")

default_ckpt = "checkpoints/best_model_mobilenetv2.pth"
ckpt_path = st.sidebar.text_input(
    "Đường dẫn checkpoint (.pth)",
    value=default_ckpt,
)

img_size = st.sidebar.number_input(
    "Kích thước resize (train dùng 512)",
    min_value=128,
    max_value=1024,
    value=512,
    step=32,
)

cls_thresh = st.sidebar.slider(
    "Ngưỡng classification (score ≥ thresh ⇒ tampered)",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
)

mask_thresh = st.sidebar.slider(
    "Ngưỡng nhị phân hoá mask pixel fake",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
)

overlay_alpha = st.sidebar.slider(
    "Độ đậm overlay mask",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
)

use_cpu = st.sidebar.checkbox("Force dùng CPU (bỏ chọn để dùng GPU nếu có)", value=False)


# ======================
# Main UI
# ======================
st.title("Image Splicing Detection & Localization (MobileNetV2+SRM)")

uploaded_file = st.file_uploader("Upload 1 ảnh (JPG/PNG/BMP...)", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])

if uploaded_file is not None:
    # Hiển thị ảnh gốc
    file_bytes = uploaded_file.read()
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    orig_w, orig_h = img.size

    st.subheader("Ảnh gốc")
    st.image(img,  width="stretch")

    # Nút chạy inference
    if st.button("Chạy phát hiện tampering"):
        # Chọn device
        if use_cpu:
            device_str = "cpu"
        else:
            device_str = "cuda" if torch.cuda.is_available() else "cpu"

        st.write(f"Thiết bị sử dụng: `{device_str}`")

        # Load model
        try:
            with st.spinner("Đang load model..."):
                model, device = load_model(ckpt_path, device_str)
        except Exception as e:
            st.error(f"Lỗi load checkpoint: {e}")
            st.stop()

        # Preprocess
        transform = build_transform(img_size)
        img_t = transform(img)  # 3xH'xW'
        img_t = img_t.unsqueeze(0).to(device)  # 1x3xH'xW'

        # Inference
        with torch.no_grad():
            outputs = model(img_t)

        # Lấy xác suất tampered
        if "det_prob" in outputs:
            det_prob = outputs["det_prob"]
        else:
            det_prob = torch.sigmoid(outputs["det_logits"])

        score = det_prob.squeeze().item()

        st.markdown(f"### Tampering score (probability): **{score:.4f}**")

        if score < cls_thresh:
            st.success("Prediction: **AUTHENTIC** (không bị tampering theo ngưỡng hiện tại).")
        else:
            st.warning("Prediction: **TAMPERED** (ảnh bị tampering theo ngưỡng hiện tại).")

        # Xử lý mask nếu tampered
        if score >= cls_thresh:
            # Lấy mask_prob
            if "mask_prob" in outputs:
                mask_prob = outputs["mask_prob"]  # 1x1xH'xW'
            else:
                mask_logits = outputs["mask_logits"]
                mask_prob = torch.sigmoid(mask_logits)

            # Upsample về size gốc
            mask_up = F.interpolate(
                mask_prob,
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            )  # 1x1xH_origxW_orig

            mask_np = mask_up.squeeze().cpu().numpy()  # HxW, 0..1
            mask_bin = (mask_np >= mask_thresh).astype(np.uint8) * 255  # 0/255
            tampered_ratio = mask_bin.mean() / 255.0

            st.write(f"Tỉ lệ pixel bị đánh dấu tampered: **{tampered_ratio*100:.2f}%**")

            # Hiển thị mask + overlay
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Mask (nhị phân)")
                st.image(mask_bin, clamp=True, caption="Mask tampered (0/255)", width="stretch")

            with col2:
                st.subheader("Overlay (mask tô đỏ)")
                overlay_img = make_overlay(img, mask_bin / 255.0, alpha=overlay_alpha)
                st.image(overlay_img,  width="stretch")


else:
    st.info("Hãy upload 1 ảnh để chạy phát hiện tampering.")
