from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import io
import math
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import joblib
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage
from scipy.spatial import ConvexHull

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
STAGE1_MODEL_PATH = MODEL_DIR / "knn_stage1_banana_detector.joblib"
STAGE2_MODEL_PATH = MODEL_DIR / "knn_stage2_ripeness_classifier.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.joblib"

MAX_IMAGE_DIMENSION: int = 320
MIN_OBJECT_AREA_RATIO: float = 0.015
MAX_OBJECT_AREA_RATIO: float = 0.985
STAGE1_CONFIDENCE_THRESHOLD: float = 0.58
ALLOWED_EXTENSIONS: Tuple[str, ...] = ("jpg", "jpeg", "png")

NON_BANANA_REJECTION_MESSAGE: str = (
    "Objek bukan pisang atau pisang tidak dapat dikenali. "
    "Silakan gunakan foto Pisang Cavendish yang jelas."
)

RIPENESS_CLASS_CODES: List[str] = [
    "mentah",
    "setengah_matang",
    "matang",
    "terlalu_matang",
]

ALL_DATASET_CLASSES: List[str] = [
    "mentah",
    "setengah_matang",
    "matang",
    "terlalu_matang",
    "bukan_pisang",
]

STAGE1_FEATURE_NAMES: List[str] = [
    "mean_h",
    "mean_s",
    "mean_v",
    "std_h",
    "std_s",
    "std_v",
    "banana_spectrum_ratio",
    "non_banana_hue_ratio",
    "glare_ratio",
    "aspect_ratio_bbox",
    "elongation_ratio",
    "circularity",
    "solidity",
    "extent",
    "texture_gradient_mean",
    "texture_local_std",
]

STAGE2_FEATURE_NAMES: List[str] = [
    "mean_h",
    "median_h",
    "q25_h",
    "q75_h",
    "mean_v",
    "std_v",
    "green_ratio",
    "yellow_green_ratio",
    "yellow_ratio",
    "brown_ratio",
    "peel_green_pct",
    "peel_yellow_pct",
]

RIPENESS_CATEGORIES: Dict[str, Dict[str, str]] = {
    "Mentah": {
        "code": "mentah",
        "label": "Pisang Mentah",
        "short_label": "Mentah",
        "full_label": "Pisang Cavendish Mentah",
        "stage": "Tahap 1 — Mentah",
        "color_hex": "#2F6B45",
        "bg_hex": "#EBF3EE",
        "hsv_profile": "Dominan hijau (Hue 28°–75°)",
        "description": "Kulit hijau segar merata, tekstur masih keras, dan belum manis.",
        "explanation": "Kulit pisang didominasi warna hijau sehingga sistem mengklasifikasikan buah ke dalam kategori mentah.",
    },
    "Setengah Matang": {
        "code": "setengah_matang",
        "label": "Pisang Setengah Matang",
        "short_label": "Setengah Matang",
        "full_label": "Pisang Cavendish Setengah Matang",
        "stage": "Tahap 2 — Setengah Matang",
        "color_hex": "#7FA332",
        "bg_hex": "#F4F8E8",
        "hsv_profile": "Hijau kekuningan (Hue 24°–30°)",
        "description": "Warna kulit bergradasi hijau kekuningan dan tekstur mulai melunak.",
        "explanation": "Kulit pisang memiliki perpaduan warna hijau dan kuning sehingga sistem mengklasifikasikan buah ke dalam kategori setengah matang.",
    },
    "Matang": {
        "code": "matang",
        "label": "Pisang Matang",
        "short_label": "Matang",
        "full_label": "Pisang Cavendish Matang",
        "stage": "Tahap 3 — Matang",
        "color_hex": "#D99B1E",
        "bg_hex": "#FDF8EA",
        "hsv_profile": "Kuning cerah (Hue 18°–24°)",
        "description": "Kulit kuning cerah merata, rasa manis optimal, dan siap dikonsumsi.",
        "explanation": "Kulit pisang didominasi warna kuning sehingga sistem mengklasifikasikan buah ke dalam kategori matang.",
    },
    "Terlalu Matang": {
        "code": "terlalu_matang",
        "label": "Pisang Terlalu Matang",
        "short_label": "Terlalu Matang",
        "full_label": "Pisang Cavendish Terlalu Matang",
        "stage": "Tahap 4 — Terlalu Matang",
        "color_hex": "#8C5A2B",
        "bg_hex": "#F7EFE8",
        "hsv_profile": "Kuning tua bercak cokelat (Hue <18°)",
        "description": "Kulit kuning tua dengan banyak bercak cokelat dan tekstur sangat lunak.",
        "explanation": "Kulit pisang kuning tua disertai bercak cokelat sehingga sistem mengklasifikasikan buah ke dalam kategori terlalu matang.",
    },
}

CODE_TO_DISPLAY: Dict[str, Dict[str, str]] = {
    item["code"]: item for item in RIPENESS_CATEGORIES.values()
}


class SegmentationError(ValueError):
    pass


def is_model_available() -> bool:
    return STAGE1_MODEL_PATH.is_file() and STAGE2_MODEL_PATH.is_file()


@lru_cache(maxsize=2)
def _load_models_by_mtime(mtime1: float, mtime2: float) -> Tuple[Any, Any]:
    return joblib.load(STAGE1_MODEL_PATH), joblib.load(STAGE2_MODEL_PATH)


def get_trained_knn_models() -> Tuple[Any, Any]:
    return _load_models_by_mtime(
        STAGE1_MODEL_PATH.stat().st_mtime,
        STAGE2_MODEL_PATH.stat().st_mtime,
    )


def _composite_rgba_to_rgb(pil_image: Image.Image) -> Image.Image:
    has_alpha = pil_image.mode in ("RGBA", "LA") or (
        pil_image.mode == "P" and "transparency" in pil_image.info
    )
    if has_alpha:
        rgba = pil_image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (248, 248, 245, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return pil_image.convert("RGB")


def load_image_from_source(
    file_source: Any,
) -> Tuple[Optional[Image.Image], Optional[Dict[str, Any]]]:
    if file_source is None:
        return None, None
    try:
        raw_bytes = file_source.getvalue()
        if not raw_bytes or len(raw_bytes) < 16:
            return None, {
                "error": "File gambar kosong atau rusak. Silakan pilih foto JPG, JPEG, atau PNG yang valid."
            }

        file_hash = hashlib.md5(raw_bytes).hexdigest()
        with Image.open(io.BytesIO(raw_bytes)) as verify_img:
            verify_img.verify()

        pil_image = Image.open(io.BytesIO(raw_bytes))
        rgb_image = _composite_rgba_to_rgb(pil_image)
        width, height = rgb_image.size
        if width < 24 or height < 24:
            return None, {
                "error": f"Resolusi gambar terlalu kecil ({width}×{height} px). Gunakan foto yang jelas."
            }

        return rgb_image, {
            "filename": getattr(file_source, "name", "camera_capture.jpg"),
            "file_hash": file_hash,
            "width": width,
            "height": height,
            "resolution": f"{width} × {height} px",
            "aspect_ratio": round(width / height, 2) if height > 0 else 0,
            "format": (pil_image.format or "PNG").upper(),
            "color_mode": pil_image.mode,
            "size_kb": round(len(raw_bytes) / 1024.0, 1),
        }
    except Exception:
        return None, {
            "error": "File gambar rusak atau format tidak dapat dibaca. Gunakan JPG, JPEG, atau PNG yang valid."
        }


def read_image_bgr(image_path: Path) -> np.ndarray:
    if not image_path.is_file():
        raise FileNotFoundError(f"File citra tidak ditemukan: {image_path}")
    with Image.open(image_path) as pil_img:
        rgb = np.asarray(_composite_rgba_to_rgb(pil_img), dtype=np.uint8)
        return rgb[..., ::-1].copy()


def pil_to_bgr(pil_image: Image.Image) -> np.ndarray:
    return np.asarray(_composite_rgba_to_rgb(pil_image), dtype=np.uint8)[..., ::-1].copy()


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    if HAS_OPENCV:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_bgr[..., ::-1].copy()


def resize_keep_aspect_ratio(
    image_bgr: np.ndarray,
    max_dim: int = MAX_IMAGE_DIMENSION,
) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    scale = float(max_dim) / float(max(height, width))
    new_w = max(32, int(round(width * scale)))
    new_h = max(32, int(round(height * scale)))

    if HAS_OPENCV:
        interp = cv2.INTER_AREA if scale <= 1.0 else cv2.INTER_LINEAR
        return cv2.resize(image_bgr, (new_w, new_h), interpolation=interp)

    pil_resized = Image.fromarray(image_bgr[..., ::-1]).resize(
        (new_w, new_h),
        Image.Resampling.BILINEAR,
    )
    return np.asarray(pil_resized, dtype=np.uint8)[..., ::-1].copy()


def apply_gaussian_blur(
    image_bgr: np.ndarray,
    kernel_size: Tuple[int, int] = (5, 5),
) -> np.ndarray:
    if HAS_OPENCV:
        return cv2.GaussianBlur(image_bgr, kernel_size, 0)
    radius = max(1.0, (kernel_size[0] - 1) / 3.5)
    blurred = Image.fromarray(image_bgr[..., ::-1]).filter(
        ImageFilter.GaussianBlur(radius=radius)
    )
    return np.asarray(blurred, dtype=np.uint8)[..., ::-1].copy()


def convert_bgr_to_hsv(image_bgr: np.ndarray) -> np.ndarray:
    if HAS_OPENCV:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    rgb = image_bgr[..., ::-1].astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cmax = np.max(rgb, axis=-1)
    cmin = np.min(rgb, axis=-1)
    delta = cmax - cmin

    hue = np.zeros_like(cmax, dtype=np.float32)
    mask = delta > 1e-6
    idx_r = mask & (cmax == r)
    idx_g = mask & (cmax == g) & (~idx_r)
    idx_b = mask & (~idx_r) & (~idx_g)

    hue[idx_r] = (60.0 * ((g[idx_r] - b[idx_r]) / delta[idx_r]) + 360.0) % 360.0
    hue[idx_g] = 60.0 * ((b[idx_g] - r[idx_g]) / delta[idx_g]) + 120.0
    hue[idx_b] = 60.0 * ((r[idx_b] - g[idx_b]) / delta[idx_b]) + 240.0

    h_cv = np.clip(np.round(hue / 2.0), 0, 179).astype(np.uint8)
    s_cv = np.clip(
        np.round(np.where(cmax > 1e-6, (delta / np.maximum(cmax, 1e-6)) * 255.0, 0.0)),
        0,
        255,
    ).astype(np.uint8)
    v_cv = np.clip(np.round(cmax * 255.0), 0, 255).astype(np.uint8)
    return np.stack([h_cv, s_cv, v_cv], axis=-1)


def segment_and_detect_contour(
    image_bgr: np.ndarray,
    image_hsv: np.ndarray,
) -> Dict[str, Any]:
    h_img, w_img = image_hsv.shape[:2]
    total_area = float(h_img * w_img)
    border_pad = max(3, min(h_img, w_img) // 25)

    border_mask = np.zeros((h_img, w_img), dtype=bool)
    border_mask[:border_pad, :] = border_mask[-border_pad:, :] = True
    border_mask[:, :border_pad] = border_mask[:, -border_pad:] = True

    bg_bgr_median = np.median(image_bgr[border_mask], axis=0)
    bgr_diff = np.linalg.norm(
        image_bgr.astype(np.float32) - bg_bgr_median.astype(np.float32),
        axis=-1,
    )
    s_chan = image_hsv[..., 1].astype(np.float32)
    v_chan = image_hsv[..., 2].astype(np.float32)
    bg_s_median = float(np.median(s_chan[border_mask]))
    bg_v_median = float(np.median(v_chan[border_mask]))

    adaptive_diff_thresh = max(
        24.0,
        float(np.percentile(bgr_diff[border_mask], 90)) * 1.25,
    )
    mask_by_saturation = (s_chan > max(30.0, bg_s_median + 16.0)) & (v_chan > 18.0)

    if bg_s_median < 25.0:
        if bg_v_median >= 65.0:
            mask_by_bg_diff = (bgr_diff > max(44.0, adaptive_diff_thresh * 1.4)) & (
                (s_chan >= 24.0) | (v_chan < bg_v_median * 0.62)
            )
        else:
            mask_by_bg_diff = (bgr_diff > max(34.0, adaptive_diff_thresh)) & (
                v_chan > bg_v_median + 32.0
            )
    else:
        mask_by_bg_diff = bgr_diff > max(32.0, adaptive_diff_thresh * 1.15)

    raw_mask = mask_by_bg_diff | mask_by_saturation
    if float(np.std(image_bgr)) < 8.0 and float(np.std(s_chan)) < 6.0:
        raise SegmentationError(
            "Segmentasi gagal: Citra terlalu polos atau kontras sangat rendah."
        )

    struct = ndimage.iterate_structure(ndimage.generate_binary_structure(2, 2), 2)
    cleaned = ndimage.binary_closing(raw_mask, structure=struct, iterations=2)
    cleaned = ndimage.binary_opening(cleaned, structure=struct, iterations=1)
    cleaned = ndimage.binary_fill_holes(cleaned)

    labeled, num_features = ndimage.label(cleaned)
    if num_features == 0:
        raise SegmentationError(
            "Segmentasi gagal: Tidak ditemukan kontur objek pada citra."
        )

    component_sizes = ndimage.sum(cleaned, labeled, range(1, num_features + 1))
    object_mask = labeled == (int(np.argmax(component_sizes)) + 1)
    contour_area = float(np.sum(object_mask))
    area_ratio = contour_area / total_area

    if area_ratio < MIN_OBJECT_AREA_RATIO or area_ratio > MAX_OBJECT_AREA_RATIO:
        raise SegmentationError(
            "Segmentasi gagal: Area objek terlalu kecil atau tidak dapat dipisahkan dari latar belakang."
        )

    eroded = ndimage.binary_erosion(object_mask)
    perimeter = max(1.0, float(np.sum(object_mask & (~eroded))) * 1.05)
    ys, xs = np.where(object_mask)
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    w_b = max(1, x_max - x_min + 1)
    h_b = max(1, y_max - y_min + 1)
    bbox_area = float(w_b * h_b)

    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    hull_area = max(contour_area, bbox_area * 0.78)
    try:
        if len(pts) >= 10:
            hull_area = max(contour_area, float(ConvexHull(pts).volume))
    except Exception:
        pass

    if len(pts) >= 5:
        eigvals = np.maximum(np.linalg.eigvalsh(np.cov(pts, rowvar=False)), 1e-3)
        elongation_ratio = float(math.sqrt(eigvals[-1] / eigvals[0]))
    else:
        elongation_ratio = float(max(w_b, h_b) / max(1, min(w_b, h_b)))

    return {
        "object_mask": object_mask,
        "contour": pts,
        "contour_area": contour_area,
        "area_ratio": area_ratio,
        "perimeter": perimeter,
        "hull_area": hull_area,
        "bbox": (x_min, y_min, w_b, h_b),
        "bbox_area": bbox_area,
        "elongation_ratio": elongation_ratio,
    }


def extract_shape_and_texture_features(
    image_hsv: np.ndarray,
    seg_info: Dict[str, Any],
) -> Dict[str, float]:
    area = float(seg_info["contour_area"])
    perimeter = float(seg_info["perimeter"])
    hull_area = float(seg_info["hull_area"])
    bbox_area = float(seg_info["bbox_area"])
    _, _, w_b, h_b = seg_info["bbox"]
    object_mask = seg_info["object_mask"]

    aspect_ratio_bbox = float(max(w_b, h_b)) / float(max(1, min(w_b, h_b)))
    circularity = min(
        1.0,
        max(0.0, float((4.0 * math.pi * area) / max(1.0, perimeter * perimeter))),
    )
    solidity = min(1.0, max(0.0, float(area / max(1.0, hull_area))))
    extent = min(1.0, max(0.0, float(area / max(1.0, bbox_area))))

    v_float = image_hsv[..., 2].astype(np.float32)
    grad_mag = np.hypot(
        ndimage.sobel(v_float, axis=1),
        ndimage.sobel(v_float, axis=0),
    )
    inner_mask = ndimage.binary_erosion(object_mask, iterations=2)
    if np.sum(inner_mask) < 20:
        inner_mask = object_mask

    local_mean = ndimage.uniform_filter(v_float, size=5)
    local_var = ndimage.uniform_filter(v_float * v_float, size=5) - local_mean * local_mean
    local_std = np.sqrt(np.maximum(0.0, local_var))

    return {
        "aspect_ratio_bbox": round(aspect_ratio_bbox, 4),
        "elongation_ratio": round(float(seg_info["elongation_ratio"]), 4),
        "circularity": round(circularity, 4),
        "solidity": round(solidity, 4),
        "extent": round(extent, 4),
        "texture_gradient_mean": round(float(np.mean(grad_mag[inner_mask])) / 255.0, 4),
        "texture_local_std": round(float(np.mean(local_std[inner_mask])) / 255.0, 4),
    }


def extract_object_hsv_color_features(
    image_hsv: np.ndarray,
    object_mask: np.ndarray,
) -> Dict[str, float]:
    eroded_mask = ndimage.binary_erosion(object_mask, iterations=2)
    active_mask = eroded_mask if int(np.sum(eroded_mask)) >= 50 else object_mask

    h_obj = image_hsv[..., 0][active_mask].astype(np.float64)
    s_obj = image_hsv[..., 1][active_mask].astype(np.float64)
    v_obj = image_hsv[..., 2][active_mask].astype(np.float64)
    if h_obj.size == 0:
        raise SegmentationError("Segmentasi gagal: Mask objek kosong.")

    ys_mask, xs_mask = np.where(active_mask)
    y_span = max(1.0, float(np.max(ys_mask) - np.min(ys_mask)))
    x_span = max(1.0, float(np.max(xs_mask) - np.min(xs_mask)))
    y_rel = (ys_mask.astype(np.float64) - float(np.min(ys_mask))) / y_span
    x_rel = (xs_mask.astype(np.float64) - float(np.min(xs_mask))) / x_span
    body_region_all = (
        (y_rel >= 0.16) & (y_rel <= 0.88) & (x_rel >= 0.10) & (x_rel <= 0.90)
    )

    valid_peel_px = (
        (v_obj >= 22.0)
        & (~((s_obj < 16.0) & (v_obj > 240.0)))
        & (~((s_obj < 24.0) & (v_obj < 45.0)))
    )
    if np.sum(valid_peel_px) >= 30:
        h_calc = h_obj[valid_peel_px]
        s_calc = s_obj[valid_peel_px]
        v_calc = v_obj[valid_peel_px]
        body_calc = body_region_all[valid_peel_px]
    else:
        h_calc, s_calc, v_calc, body_calc = h_obj, s_obj, v_obj, body_region_all

    n_px = float(h_calc.size)
    n_total_obj = float(h_obj.size)

    bright_lemon_yellow = (
        (h_calc >= 24.5) & (h_calc <= 30.5) & (s_calc >= 30.0) & (v_calc >= 195.0)
    )
    green_only_px = (
        (~bright_lemon_yellow)
        & (h_calc >= 28.0)
        & (h_calc <= 85.0)
        & (s_calc >= 28.0)
        & (v_calc >= 28.0)
    )
    yellow_green_px = (
        (~bright_lemon_yellow)
        & (~green_only_px)
        & (h_calc >= 24.5)
        & (h_calc < 28.0)
        & (s_calc >= 28.0)
        & (v_calc >= 35.0)
    )
    green_all_px = green_only_px | yellow_green_px

    raw_brown_cond = (
        ((h_calc >= 2.0) & (h_calc <= 20.5) & (s_calc >= 30.0) & (v_calc >= 18.0) & (v_calc < 138.0))
        | ((h_calc > 20.5) & (h_calc <= 23.5) & (s_calc >= 30.0) & (v_calc >= 18.0) & (v_calc < 105.0))
        | ((v_calc >= 18.0) & (v_calc < 72.0) & (s_calc >= 22.0))
        | ((h_calc >= 2.0) & (h_calc < 14.5) & (s_calc >= 38.0) & (v_calc >= 18.0) & (v_calc < 155.0))
    )
    brown_px = (~green_all_px) & (~bright_lemon_yellow) & body_calc & raw_brown_cond
    stem_tip_dark_px = (
        (~green_all_px) & (~bright_lemon_yellow) & (~body_calc) & raw_brown_cond
    )
    yellow_px = (
        (~green_all_px)
        & (~brown_px)
        & (~stem_tip_dark_px)
        & (
            bright_lemon_yellow
            | ((h_calc >= 12.0) & (h_calc < 25.0) & (s_calc >= 24.0) & (v_calc >= 90.0))
            | ((h_calc >= 12.0) & (h_calc <= 29.0) & (s_calc >= 14.0) & (v_calc >= 145.0))
        )
    )

    peel_mask = ~stem_tip_dark_px
    n_peel = float(max(1, int(np.sum(peel_mask))))
    other_px = peel_mask & (~green_all_px) & (~brown_px) & (~yellow_px)

    green_ratio = float(np.sum(green_only_px & peel_mask) / n_peel)
    yellow_green_ratio = float(np.sum(yellow_green_px & peel_mask) / n_peel)
    yellow_ratio = float(np.sum(yellow_px & peel_mask) / n_peel)
    brown_ratio = float(np.sum(brown_px & peel_mask) / n_peel)
    body_brown_ratio = float(
        np.sum(brown_px & body_calc) / float(max(1, int(np.sum(body_calc))))
    )

    peel_green_pct = round(float(np.sum(green_all_px & peel_mask) / n_peel) * 100.0, 1)
    peel_yellow_pct = round(float(np.sum(yellow_px & peel_mask) / n_peel) * 100.0, 1)
    peel_brown_pct = round(float(np.sum(brown_px & peel_mask) / n_peel) * 100.0, 1)
    peel_other_pct = round(
        max(0.0, 100.0 - (peel_green_pct + peel_yellow_pct + peel_brown_pct)),
        1,
    )

    banana_spectrum_px = green_all_px | yellow_px | brown_px
    banana_spectrum_ratio = float(np.sum(banana_spectrum_px) / n_px)
    non_banana_hue_ratio = float(
        np.sum(((h_calc < 2.0) | (h_calc > 86.0)) & (s_calc >= 38.0)) / n_px
    )
    glare_ratio = float(np.sum((s_obj < 22.0) & (v_obj > 235.0)) / n_total_obj)

    if np.sum(banana_spectrum_px) >= 30:
        h_stat = h_calc[banana_spectrum_px]
        s_stat = s_calc[banana_spectrum_px]
        v_stat = v_calc[banana_spectrum_px]
    else:
        h_stat, s_stat, v_stat = h_calc, s_calc, v_calc

    h_low, h_high = np.percentile(h_stat, [10, 90])
    v_low, v_high = np.percentile(v_stat, [10, 90])
    core_mask = (
        (h_stat >= h_low) & (h_stat <= h_high) & (v_stat >= v_low) & (v_stat <= v_high)
    )
    if np.sum(core_mask) >= 20:
        h_core, s_core, v_core = h_stat[core_mask], s_stat[core_mask], v_stat[core_mask]
    else:
        h_core, s_core, v_core = h_stat, s_stat, v_stat

    q25_h, median_h, q75_h = [float(x) for x in np.percentile(h_core, [25, 50, 75])]
    return {
        "mean_h": round(float(np.mean(h_core)), 4),
        "mean_s": round(float(np.mean(s_core)), 4),
        "mean_v": round(float(np.clip(np.mean(v_core), 85.0, 180.0)), 4),
        "std_h": round(float(np.std(h_core)), 4),
        "std_s": round(float(np.std(s_core)), 4),
        "std_v": round(float(np.std(v_core)), 4),
        "median_h": round(median_h, 4),
        "q25_h": round(q25_h, 4),
        "q75_h": round(q75_h, 4),
        "iqr_h": round(float(q75_h - q25_h), 4),
        "green_ratio": round(green_ratio, 4),
        "yellow_green_ratio": round(yellow_green_ratio, 4),
        "yellow_ratio": round(yellow_ratio, 4),
        "brown_ratio": round(brown_ratio, 4),
        "body_brown_ratio": round(body_brown_ratio, 4),
        "banana_spectrum_ratio": round(banana_spectrum_ratio, 4),
        "non_banana_hue_ratio": round(non_banana_hue_ratio, 4),
        "glare_ratio": round(glare_ratio, 4),
        "peel_green_pct": peel_green_pct,
        "peel_yellow_pct": peel_yellow_pct,
        "peel_brown_pct": peel_brown_pct,
        "peel_other_pct": peel_other_pct,
        "raw_other_pct": round(float(np.sum(other_px) / n_peel) * 100.0, 1),
    }


def determine_dominant_color_info(
    image_rgb: np.ndarray,
    object_mask: np.ndarray,
    color_feats: Dict[str, float],
) -> Dict[str, str]:
    obj_rgb = image_rgb[object_mask]
    if obj_rgb.size > 0:
        r_med, g_med, b_med = [int(round(x)) for x in np.median(obj_rgb, axis=0)]
    else:
        r_med, g_med, b_med = (47, 107, 69)
    swatch_hex = f"#{r_med:02X}{g_med:02X}{b_med:02X}"

    mean_h = color_feats["mean_h"]
    green_r = color_feats["green_ratio"]
    yg_r = color_feats["yellow_green_ratio"]
    yellow_r = color_feats["yellow_ratio"]
    brown_r = color_feats["brown_ratio"]
    non_b = color_feats["non_banana_hue_ratio"]

    if non_b > 0.40:
        dom_name = (
            "Biru / Ungu / Non-Spektrum Pisang"
            if mean_h > 95
            else "Merah / Non-Spektrum Pisang"
        )
    elif brown_r >= 0.12:
        dom_name = "Kuning Tua Berbintik Cokelat"
    elif yg_r >= 0.22 or (24.5 <= mean_h < 29.0 and (green_r + yg_r) >= 0.30):
        dom_name = "Hijau Kekuningan"
    elif (green_r + yg_r) >= yellow_r and (green_r + yg_r) >= 0.35:
        dom_name = "Hijau Segar Dominan"
    elif yellow_r >= 0.25:
        dom_name = "Kuning Cerah Dominan"
    elif 28.0 <= mean_h <= 85.0:
        dom_name = "Hijau"
    elif 24.5 <= mean_h < 28.0:
        dom_name = "Hijau Kekuningan"
    elif 15.0 <= mean_h < 24.5:
        dom_name = "Kuning"
    else:
        dom_name = "Cokelat / Campuran"

    return {
        "dominant_color_name": dom_name,
        "dominant_color_hex": swatch_hex,
    }


def build_ripeness_explanation(
    predicted_code: str,
    features_dict: Dict[str, float],
) -> str:
    g_pct = float(features_dict.get("peel_green_pct", 0.0))
    y_pct = float(features_dict.get("peel_yellow_pct", 0.0))
    b_pct = float(features_dict.get("peel_brown_pct", 0.0))

    if predicted_code == "mentah":
        return (
            f"Kulit pisang didominasi warna hijau ({g_pct:.1f}%) sehingga sistem "
            "mengklasifikasikan buah ke dalam kategori mentah."
        )
    if predicted_code == "setengah_matang":
        return (
            f"Kulit pisang memiliki perpaduan warna hijau ({g_pct:.1f}%) dan kuning "
            f"({y_pct:.1f}%) sehingga sistem mengklasifikasikan buah ke dalam kategori setengah matang."
        )
    if predicted_code == "matang":
        return (
            f"Kulit pisang didominasi warna kuning ({y_pct:.1f}%) sehingga sistem "
            "mengklasifikasikan buah ke dalam kategori matang."
        )
    if predicted_code == "terlalu_matang":
        return (
            f"Kulit pisang berwarna kuning tua ({y_pct:.1f}%) disertai bercak cokelat "
            f"({b_pct:.1f}%) sehingga sistem mengklasifikasikan buah ke dalam kategori terlalu matang."
        )
    return "Sistem mengklasifikasikan tingkat kematangan buah berdasarkan analisis warna kulit pisang."


def build_feature_vectors(
    color_feats: Dict[str, float],
    shape_feats: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    merged: Dict[str, float] = {**color_feats, **shape_feats}
    vec_stage1 = np.array([float(merged[k]) for k in STAGE1_FEATURE_NAMES], dtype=np.float64)
    vec_stage2 = np.array([float(merged[k]) for k in STAGE2_FEATURE_NAMES], dtype=np.float64)
    return vec_stage1, vec_stage2, merged


def process_and_extract_from_bgr(image_bgr: np.ndarray) -> Dict[str, Any]:
    resized_bgr = resize_keep_aspect_ratio(image_bgr, MAX_IMAGE_DIMENSION)
    blurred_bgr = apply_gaussian_blur(resized_bgr, (5, 5))
    image_hsv = convert_bgr_to_hsv(blurred_bgr)
    seg_info = segment_and_detect_contour(blurred_bgr, image_hsv)
    object_mask = seg_info["object_mask"]

    shape_feats = extract_shape_and_texture_features(image_hsv, seg_info)
    color_feats = extract_object_hsv_color_features(image_hsv, object_mask)
    vec_stage1, vec_stage2, all_features = build_feature_vectors(color_feats, shape_feats)
    vis_rgb = bgr_to_rgb(resized_bgr)
    dom_color_info = determine_dominant_color_info(vis_rgb, object_mask, color_feats)

    return {
        "resized_rgb": vis_rgb,
        "object_mask": object_mask,
        "area_ratio": round(float(seg_info["area_ratio"]), 4),
        "vec_stage1": vec_stage1,
        "vec_stage2": vec_stage2,
        "features": all_features,
        "dominant_color_name": dom_color_info["dominant_color_name"],
        "dominant_color_hex": dom_color_info["dominant_color_hex"],
        "peel_color_percentages": {
            "green": color_feats["peel_green_pct"],
            "yellow": color_feats["peel_yellow_pct"],
            "brown": color_feats["peel_brown_pct"],
            "other": color_feats["peel_other_pct"],
        },
    }


def extract_both_stage_features_from_file(
    image_path: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    processed = process_and_extract_from_bgr(read_image_bgr(image_path))
    return processed["vec_stage1"], processed["vec_stage2"], processed["features"]


def predict_two_stage_from_vectors(
    vec_stage1: np.ndarray,
    vec_stage2: np.ndarray,
    features_dict: Dict[str, float],
) -> Dict[str, Any]:
    if not is_model_available():
        return {
            "model_trained": False,
            "is_banana": None,
            "object_status": "Menunggu Pelatihan Model KNN",
            "stage1_prediction": None,
            "stage1_confidence": None,
            "predicted_code": None,
            "predicted_info": None,
            "stage2_confidence": None,
            "stage2_probabilities": {},
            "explanation_text": None,
            "status_message": "Model belum dilatih",
        }

    stage1_pipeline, stage2_pipeline = get_trained_knn_models()
    x1 = vec_stage1.reshape(1, -1)
    pred1 = str(stage1_pipeline.predict(x1)[0])
    banana_prob = 1.0 if pred1 == "pisang" else 0.0
    classes1 = [str(c) for c in getattr(stage1_pipeline, "classes_", ["pisang"])]

    if len(classes1) >= 2 and hasattr(stage1_pipeline, "predict_proba"):
        proba1 = stage1_pipeline.predict_proba(x1)[0]
        if "pisang" in classes1:
            banana_prob = float(proba1[classes1.index("pisang")])
    elif (
        len(classes1) == 1
        and "scaler" in stage1_pipeline.named_steps
        and "knn" in stage1_pipeline.named_steps
    ):
        scaler1 = stage1_pipeline.named_steps["scaler"]
        knn1 = stage1_pipeline.named_steps["knn"]
        min_scales = np.array(
            [6.0, 28.0, 32.0, 6.0, 28.0, 20.0, 0.18, 0.12, 0.10, 1.20, 1.35, 0.25, 0.18, 0.18, 0.08, 0.03]
        )
        eff_scale = np.maximum(scaler1.scale_, min_scales)
        x1_norm = (x1 - scaler1.mean_) / eff_scale
        train_norm = knn1._fit_X * (scaler1.scale_ / eff_scale)
        dists = np.linalg.norm(train_norm - x1_norm, axis=1)
        mean_dist = float(np.mean(np.sort(dists)[: min(5, len(dists))]))
        banana_prob = float(np.clip(math.exp(-max(0.0, mean_dist - 2.2) / 5.5), 0.0, 1.0))

    non_banana_hue = float(features_dict.get("non_banana_hue_ratio", 0.0))
    banana_spectrum = float(features_dict.get("banana_spectrum_ratio", 1.0))
    elongation = float(features_dict.get("elongation_ratio", 2.0))
    circularity = float(features_dict.get("circularity", 0.4))
    solidity = float(features_dict.get("solidity", 0.85))

    plausible_color = (non_banana_hue < 0.35) and (banana_spectrum >= 0.45)
    plausible_shape = not (
        (elongation < 1.28) and (circularity >= 0.74) and (solidity >= 0.91)
    )

    if not (
        (pred1 == "pisang")
        and (banana_prob >= STAGE1_CONFIDENCE_THRESHOLD)
        and plausible_color
        and plausible_shape
    ):
        return {
            "model_trained": True,
            "is_banana": False,
            "object_status": "Bukan Pisang / Objek Tidak Dikenali",
            "stage1_prediction": "bukan_pisang",
            "stage1_confidence": round(
                (1.0 - banana_prob) * 100.0 if pred1 != "pisang" else banana_prob * 100.0,
                2,
            ),
            "banana_probability": round(banana_prob * 100.0, 2),
            "predicted_code": "bukan_pisang",
            "predicted_info": None,
            "stage2_confidence": None,
            "stage2_probabilities": {},
            "explanation_text": None,
            "status_message": NON_BANANA_REJECTION_MESSAGE,
        }

    x2 = vec_stage2.reshape(1, -1)
    pred2 = str(stage2_pipeline.predict(x2)[0])
    probs2_dict: Dict[str, float] = {}
    stage2_conf = 100.0
    if hasattr(stage2_pipeline, "predict_proba"):
        proba2 = stage2_pipeline.predict_proba(x2)[0]
        probs2_dict = {
            str(lbl): round(float(p) * 100.0, 2)
            for lbl, p in zip(stage2_pipeline.classes_, proba2)
        }
        stage2_conf = round(float(np.max(proba2)) * 100.0, 2)

    return {
        "model_trained": True,
        "is_banana": True,
        "object_status": "Teridentifikasi sebagai Buah Pisang Cavendish",
        "stage1_prediction": "pisang",
        "stage1_confidence": round(banana_prob * 100.0, 2),
        "banana_probability": round(banana_prob * 100.0, 2),
        "predicted_code": pred2,
        "predicted_info": CODE_TO_DISPLAY.get(pred2),
        "stage2_confidence": stage2_conf,
        "stage2_probabilities": probs2_dict,
        "explanation_text": build_ripeness_explanation(pred2, features_dict),
        "status_message": "Klasifikasi tingkat kematangan berhasil.",
    }


def analyze_pil_image(pil_image: Image.Image) -> Dict[str, Any]:
    try:
        proc = process_and_extract_from_bgr(pil_to_bgr(pil_image))
    except SegmentationError as seg_err:
        return {
            "segmentation_success": False,
            "error_message": str(seg_err),
            "model_trained": is_model_available(),
        }
    except Exception as exc:
        return {
            "segmentation_success": False,
            "error_message": f"Gagal memproses citra: {exc}",
            "model_trained": is_model_available(),
        }

    prediction = predict_two_stage_from_vectors(
        proc["vec_stage1"],
        proc["vec_stage2"],
        proc["features"],
    )
    return {
        "segmentation_success": True,
        "error_message": None,
        "resized_rgb": proc["resized_rgb"],
        "area_ratio": proc["area_ratio"],
        "features": proc["features"],
        "dominant_color_name": proc["dominant_color_name"],
        "dominant_color_hex": proc["dominant_color_hex"],
        "peel_color_percentages": proc["peel_color_percentages"],
        **prediction,
    }
