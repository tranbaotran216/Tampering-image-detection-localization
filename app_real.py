# app_real.py
import os
import io
import tempfile
import requests

import numpy as np
from PIL import Image

import streamlit as st

import torch
import torch.nn.functional as F
import torchvision.transforms as T

# --- CẤU HÌNH API KEY TẠI ĐÂY ---
# Hãy dán API Key của bạn vào giữa cặp ngoặc kép bên dưới
SERPAPI_KEY = "API_KEY_HERE"  # <-- DÁN_KEY_VÀO_ĐÂY -->
# Ví dụ: SERPAPI_KEY = "a1b2c3d4e5..."

try:
    from models.swint2 import SwinT2_SRM_FPN_DetLoc
except ImportError:
    st.error("Lỗi: Không tìm thấy file 'models/swint2.py'. Vui lòng kiểm tra lại cấu trúc thư mục.")
    st.stop()


st.set_page_config(
    page_title="Splicing Detection & Search",
    layout="wide",
)


def build_transform(img_size: int):
    return T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ]
    )


@st.cache_resource(show_spinner=False)
def load_model(
    ckpt_path: str,
    device_str: str,
    backbone_name: str,
    use_srm: bool,
    fpn_channels: int,
    det_hidden_dim: int,
    det_dropout: float,
    det_use_p5: bool,
):
    device = torch.device(device_str)

    model = SwinT2_SRM_FPN_DetLoc(
        backbone_name=backbone_name,
        pretrained_backbone=False,
        use_srm=use_srm,
        fpn_channels=fpn_channels,
        det_hidden_dim=det_hidden_dim,
        det_dropout=det_dropout,
        det_use_p5=det_use_p5,
        img_size=None,
    ).to(device)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint không tồn tại: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, device


def make_overlay(original: Image.Image, mask_np: np.ndarray, alpha: float = 0.5):
    img_np = np.array(original).astype(np.float32)
    if img_np.max() <= 1.0:
        img_np *= 255.0
    img_np = img_np / 255.0

    m = mask_np.astype(np.float32)
    if m.max() > 1.0:
        m = m / 255.0
    m = np.clip(m, 0.0, 1.0)

    h, w = m.shape
    color = np.zeros((h, w, 3), dtype=np.float32)
    color[..., 0] = 1.0

    m_exp = m[..., None]
    overlay = img_np * (1.0 - alpha * m_exp) + color * (alpha * m_exp)
    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
    return overlay


# --- SIDEBAR CONFIG ---
st.sidebar.header("Cấu hình Model")

default_ckpt = "checkpoints/best_swintv2.pth"
ckpt_path = st.sidebar.text_input("Đường dẫn checkpoint (.pth)", value=default_ckpt)

img_size = st.sidebar.number_input(
    "Kích thước resize (256 hoặc 512)",
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

use_cpu = st.sidebar.checkbox("Force dùng CPU", value=False)

st.sidebar.subheader("Tham số mạng")
backbone_name = st.sidebar.text_input("backbone_name", value="swinv2_tiny_window8_256")
use_srm = st.sidebar.checkbox("use_srm", value=True)
fpn_channels = st.sidebar.number_input("fpn_channels", min_value=64, max_value=512, value=256, step=32)
det_hidden_dim = st.sidebar.number_input("det_hidden_dim", min_value=64, max_value=1024, value=512, step=64)
det_dropout = st.sidebar.slider("det_dropout", min_value=0.0, max_value=0.9, value=0.30, step=0.05)
det_use_p5 = st.sidebar.checkbox("det_use_p5", value=True)


# --- MAIN APP ---
st.title("Image Splicing Detection & Source Verification")

uploaded_file = st.file_uploader(
    "Upload 1 ảnh (JPG/PNG/BMP...)",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
)

if uploaded_file is None:
    st.info("Hãy upload 1 ảnh để bắt đầu.")
    st.stop()


file_bytes = uploaded_file.read()
img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
orig_w, orig_h = img.size

st.subheader("Ảnh gốc")
st.image(img, width="stretch")

# --- PHẦN 1: DETECTION ---
st.markdown("---")
st.header("1. Kiểm tra chỉnh sửa (Splicing Detection)")

if st.button("Chạy phát hiện Tampering"):
    device_str = "cpu" if use_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    st.write(f"Thiết bị sử dụng: `{device_str}`")

    try:
        with st.spinner("Đang load model..."):
            model, device = load_model(
                ckpt_path=ckpt_path,
                device_str=device_str,
                backbone_name=backbone_name,
                use_srm=use_srm,
                fpn_channels=int(fpn_channels),
                det_hidden_dim=int(det_hidden_dim),
                det_dropout=float(det_dropout),
                det_use_p5=det_use_p5,
            )
    except Exception as e:
        st.error(f"Lỗi load checkpoint/model: {e}")
        st.stop()

    transform = build_transform(int(img_size))
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_t)

    det_prob = outputs["det_prob"] if "det_prob" in outputs else torch.sigmoid(outputs["det_logits"])
    score = float(det_prob.squeeze().item())

    st.markdown(f"### Tampering score (probability): **{score:.4f}**")

    if score < cls_thresh:
        st.success("Prediction: **AUTHENTIC** (Ảnh thật)")
    else:
        st.warning("Prediction: **TAMPERED** (Ảnh đã qua chỉnh sửa)")

    # Luôn hiển thị mask để tham khảo
    if score >= cls_thresh:
        mask_prob = outputs["mask_prob"] if "mask_prob" in outputs else torch.sigmoid(outputs["mask_logits"])

        mask_up = F.interpolate(
            mask_prob,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )

        mask_np = mask_up.squeeze().detach().cpu().numpy()
        mask_bin = (mask_np >= mask_thresh).astype(np.uint8) * 255
        tampered_ratio = float(mask_bin.mean() / 255.0)

        st.write(f"Tỉ lệ pixel bị đánh dấu tampered: **{tampered_ratio * 100:.2f}%**")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Mask (nhị phân)")
            st.image(mask_bin, clamp=True, caption="Khu vực nghi ngờ", width="stretch")
        with col2:
            st.subheader("Overlay (Vùng đỏ là fake)")
            overlay_img = make_overlay(img, mask_bin / 255.0, alpha=float(overlay_alpha))
            st.image(overlay_img, width="stretch")

# --- PHẦN 2: SEARCH GOOGLE LENS (FIX CỨNG API & CHỈ HIỆN TOP 1) ---
st.markdown("---")
st.header("2. Tìm nguồn gốc (Top 1 Google Lens)")

if st.button("🔍 Search ảnh này trên Google"):
    # Kiểm tra Key cứng
    if "DÁN_KEY" in SERPAPI_KEY or not SERPAPI_KEY:
        st.error("⚠️ Bạn chưa điền API Key vào trong code! Hãy sửa dòng `SERPAPI_KEY = ...` ở đầu file.")
    else:
        with st.spinner("Đang xử lý... (1. Upload -> 2. Search Top 1)"):
            try:
                # 1. Lưu file tạm
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                    img.save(tmp_file, format="JPEG")
                    tmp_path = tmp_file.name

                # 2. Upload lên tmpfiles.org
                upload_url = "https://tmpfiles.org/api/v1/upload"
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                
                with open(tmp_path, 'rb') as f:
                    response_upload = requests.post(upload_url, files={'file': f}, headers=headers)

                if response_upload.status_code != 200:
                    st.error(f"Lỗi Upload ảnh: {response_upload.text}")
                    st.stop()
                
                try:
                    upload_data = response_upload.json()
                except ValueError:
                    st.error("Lỗi Server Upload không trả về JSON.")
                    st.stop()

                initial_url = upload_data.get("data", {}).get("url")
                if not initial_url:
                    st.error("Không lấy được link ảnh.")
                    st.stop()
                
                # Chuyển thành direct link
                hosted_image_url = initial_url.replace("tmpfiles.org/", "tmpfiles.org/dl/")
                
                # 3. Gọi SerpApi
                search_url = "https://serpapi.com/search"
                params = {
                    "engine": "google_lens",
                    "url": hosted_image_url,
                    "api_key": SERPAPI_KEY, # Dùng key cứng
                    "hl": "vi",
                    "country": "vn"
                }
                
                response_search = requests.get(search_url, params=params)
                
                # Xóa file tạm
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

                # 4. Hiển thị kết quả TOP 1
                if response_search.status_code != 200:
                    st.error(f"Lỗi SerpApi: {response_search.text}")
                else:
                    results = response_search.json()
                    
                    if "error" in results:
                        st.error(f"SerpApi báo lỗi: {results['error']}")
                    else:
                        visual_matches = results.get("visual_matches", [])
                        
                        if not visual_matches:
                            st.warning("Google không tìm thấy ảnh nào tương tự.")
                        else:
                            # --- CHỈ LẤY KẾT QUẢ ĐẦU TIÊN (TOP 1) ---
                            top1 = visual_matches[0]
                            
                            st.success("✅ Đã tìm thấy kết quả phù hợp nhất!")
                            
                            # Hiển thị Top 1 thật nổi bật
                            st.subheader(f"Top 1: {top1.get('title', 'Không có tiêu đề')}")
                            
                            col_img, col_info = st.columns([1, 2])
                            
                            with col_img:
                                thumb = top1.get("thumbnail")
                                if thumb:
                                    st.image(thumb, width=300, caption="Ảnh kết quả")
                            
                            with col_info:
                                link = top1.get("link")
                                source = top1.get("source", "N/A")
                                
                                st.markdown(f"**Nguồn:** {source}")
                                st.markdown(f"**Link gốc:** [Truy cập trang web]({link})")
                                
                                # Nếu có kích thước ảnh
                                if "img_resolution" in top1:
                                    st.text(f"Độ phân giải: {top1['img_resolution']}")

            except Exception as e:
                st.error(f"Lỗi Code Python: {str(e)}")
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.remove(tmp_path)