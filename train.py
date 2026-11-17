from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from image_processing import (
    ALL_DATASET_CLASSES,
    ALLOWED_EXTENSIONS,
    BASE_DIR,
    METADATA_PATH,
    MODEL_DIR,
    RIPENESS_CLASS_CODES,
    STAGE1_FEATURE_NAMES,
    STAGE1_MODEL_PATH,
    STAGE2_FEATURE_NAMES,
    STAGE2_MODEL_PATH,
    SegmentationError,
    extract_both_stage_features_from_file,
)

TRAINING_DATA_DIR = BASE_DIR / "dataset" / "training"
MIN_SAMPLES_PER_CLASS = 3


def load_training_dataset(dataset_dir: Path = TRAINING_DATA_DIR) -> Dict[str, Any]:
    x_stage1_list: List[np.ndarray] = []
    y_stage1_list: List[str] = []
    x_stage2_list: List[np.ndarray] = []
    y_stage2_list: List[str] = []

    class_counts: Dict[str, int] = {cls: 0 for cls in ALL_DATASET_CLASSES}
    skipped_files: List[Tuple[str, str]] = []

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Direktori dataset training tidak ditemukan: {dataset_dir}"
        )

    allowed_exts = {f".{ext.lower()}" for ext in ALLOWED_EXTENSIONS}

    for cls_name in ALL_DATASET_CLASSES:
        cls_folder = dataset_dir / cls_name
        if not cls_folder.exists():
            continue

        image_files = sorted(
            p
            for p in cls_folder.iterdir()
            if p.is_file() and p.suffix.lower() in allowed_exts
        )

        for img_path in image_files:
            try:
                vec1, vec2, _ = extract_both_stage_features_from_file(img_path)
            except SegmentationError as seg_err:
                skipped_files.append((str(img_path.relative_to(BASE_DIR)), str(seg_err)))
                continue
            except Exception as exc:
                skipped_files.append((str(img_path.relative_to(BASE_DIR)), f"Error: {exc}"))
                continue

            class_counts[cls_name] += 1
            binary_label = "bukan_pisang" if cls_name == "bukan_pisang" else "pisang"
            x_stage1_list.append(vec1)
            y_stage1_list.append(binary_label)

            if cls_name in RIPENESS_CLASS_CODES:
                x_stage2_list.append(vec2)
                y_stage2_list.append(cls_name)

    x_stage1 = (
        np.array(x_stage1_list, dtype=np.float64)
        if x_stage1_list
        else np.empty((0, len(STAGE1_FEATURE_NAMES)))
    )
    y_stage1 = np.array(y_stage1_list, dtype=str)
    x_stage2 = (
        np.array(x_stage2_list, dtype=np.float64)
        if x_stage2_list
        else np.empty((0, len(STAGE2_FEATURE_NAMES)))
    )
    y_stage2 = np.array(y_stage2_list, dtype=str)

    return {
        "x_stage1": x_stage1,
        "y_stage1": y_stage1,
        "x_stage2": x_stage2,
        "y_stage2": y_stage2,
        "class_counts": class_counts,
        "skipped_files": skipped_files,
    }


def validate_dataset_readiness(class_counts: Dict[str, int]) -> Tuple[bool, List[str]]:
    messages: List[str] = []
    for cls_name in RIPENESS_CLASS_CODES:
        count = class_counts.get(cls_name, 0)
        if count < MIN_SAMPLES_PER_CLASS:
            messages.append(
                f"  - Kelas '{cls_name}': baru tersedia {count} citra valid "
                f"(minimum {MIN_SAMPLES_PER_CLASS} citra)."
            )
    return (len(messages) == 0), messages


def train_knn_with_validation(
    features: np.ndarray,
    labels: np.ndarray,
    stage_title: str,
) -> Tuple[Pipeline, Dict[str, Any]]:
    unique_classes, counts = np.unique(labels, return_counts=True)
    min_class_samples = int(np.min(counts))
    total_samples = int(len(labels))

    base_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(metric="minkowski", p=2)),
        ]
    )

    if len(unique_classes) >= 2 and min_class_samples >= 3 and total_samples >= 10:
        n_splits = min(5, min_class_samples)
        max_train_fold_size = int(total_samples * (n_splits - 1) / n_splits)
        candidate_k = [
            k for k in (1, 3, 5, 7, 9, 11) if k <= max(1, max_train_fold_size)
        ]
        if not candidate_k:
            candidate_k = [1, 3]

        param_grid = {
            "knn__n_neighbors": candidate_k,
            "knn__weights": ["uniform", "distance"],
            "knn__metric": ["euclidean", "manhattan"],
        }

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        grid = GridSearchCV(
            estimator=base_pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="f1_macro",
            n_jobs=1,
        )
        grid.fit(features, labels)

        best_pipeline: Pipeline = grid.best_estimator_
        best_params = grid.best_params_
        cv_score = float(grid.best_score_)
        validation_method = f"{n_splits}-Fold Stratified Cross-Validation"
    else:
        k_fallback = min(3, max(1, total_samples // len(unique_classes)))
        if k_fallback % 2 == 0 and k_fallback > 1:
            k_fallback -= 1
        best_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "knn",
                    KNeighborsClassifier(
                        n_neighbors=k_fallback,
                        weights="distance",
                        metric="euclidean",
                    ),
                ),
            ]
        )
        best_pipeline.fit(features, labels)
        best_params = {
            "knn__n_neighbors": k_fallback,
            "knn__weights": "distance",
            "knn__metric": "euclidean",
        }
        cv_score = float(accuracy_score(labels, best_pipeline.predict(features)))
        validation_method = "Direct Training Fit (Data minimal)"

    train_preds = best_pipeline.predict(features)
    train_acc = float(accuracy_score(labels, train_preds))

    info = {
        "stage_title": stage_title,
        "total_samples": total_samples,
        "classes": [str(c) for c in unique_classes],
        "best_k": int(best_params["knn__n_neighbors"]),
        "best_weights": str(best_params["knn__weights"]),
        "best_metric": str(best_params["knn__metric"]),
        "validation_method": validation_method,
        "cv_f1_macro": round(cv_score * 100.0, 2),
        "train_accuracy": round(train_acc * 100.0, 2),
    }
    return best_pipeline, info


def save_trained_models(
    stage1_pipeline: Pipeline,
    stage2_pipeline: Pipeline,
    metadata: Dict[str, Any],
) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(stage1_pipeline, STAGE1_MODEL_PATH)
    joblib.dump(stage2_pipeline, STAGE2_MODEL_PATH)
    joblib.dump(metadata, METADATA_PATH)


def run_training_pipeline() -> bool:
    print("=" * 76)
    print("CAVENDISH VISION — PELATIHAN MODEL KNN DUA TAHAP (HSV & GEOMETRI BENTUK)")
    print("=" * 76)
    print(f"Direktori Dataset Training : {TRAINING_DATA_DIR}")

    data = load_training_dataset(TRAINING_DATA_DIR)
    class_counts: Dict[str, int] = data["class_counts"]
    skipped_files: List[Tuple[str, str]] = data["skipped_files"]

    print("\n[1] Rekapitulasi Jumlah Citra Training Valid per Kelas:")
    print("-" * 76)
    total_valid = 0
    for cls_name in ALL_DATASET_CLASSES:
        cnt = class_counts[cls_name]
        total_valid += cnt
        print(f"  • {cls_name:<18} : {cnt} citra")
    print("-" * 76)
    print(f"  TOTAL CITRA VALID    : {total_valid} citra")

    if skipped_files:
        print(f"  CITRA DILEWATI       : {len(skipped_files)} citra (gagal segmentasi / rusak)")

    is_ready, missing_msgs = validate_dataset_readiness(class_counts)
    if not is_ready:
        print("\n[PERINGATAN: DATASET TRAINING BELUM MENCUKUPI]")
        print("Pelatihan model KNN dibatalkan karena kelas berikut belum memenuhi jumlah minimum:")
        for msg in missing_msgs:
            print(msg)
        print("\nSilakan lengkapi foto asli pada folder dataset/training/ terlebih dahulu.")
        print("Sistem TIDAK membuat data atau model palsu.")
        print("=" * 76)
        return False

    print("\n[2] Melatih Model KNN Tahap 1 (Identifikasi Pisang vs Bukan Pisang)...")
    stage1_pipeline, stage1_info = train_knn_with_validation(
        data["x_stage1"],
        data["y_stage1"],
        stage_title="Tahap 1 — Identifikasi Pisang vs Bukan Pisang",
    )
    print(f"  • Jumlah Sampel Latih : {stage1_info['total_samples']} citra")
    print(f"  • Metode Validasi     : {stage1_info['validation_method']}")
    print(
        f"  • Nilai K Terbaik     : K = {stage1_info['best_k']} "
        f"(weights='{stage1_info['best_weights']}', metric='{stage1_info['best_metric']}')"
    )
    print(f"  • Skor CV (F1-Macro)  : {stage1_info['cv_f1_macro']:.2f}%")
    print(f"  • Akurasi Data Latih  : {stage1_info['train_accuracy']:.2f}%")

    print("\n[3] Melatih Model KNN Tahap 2 (Klasifikasi Tingkat Kematangan Pisang Cavendish)...")
    stage2_pipeline, stage2_info = train_knn_with_validation(
        data["x_stage2"],
        data["y_stage2"],
        stage_title="Tahap 2 — Klasifikasi 4 Tingkat Kematangan",
    )
    print(f"  • Jumlah Sampel Latih : {stage2_info['total_samples']} citra (khusus 4 kelas Pisang Cavendish)")
    print(f"  • Metode Validasi     : {stage2_info['validation_method']}")
    print(
        f"  • Nilai K Terbaik     : K = {stage2_info['best_k']} "
        f"(weights='{stage2_info['best_weights']}', metric='{stage2_info['best_metric']}')"
    )
    print(f"  • Skor CV (F1-Macro)  : {stage2_info['cv_f1_macro']:.2f}%")
    print(f"  • Akurasi Data Latih  : {stage2_info['train_accuracy']:.2f}%")

    metadata = {
        "trained_at": datetime.now().isoformat(),
        "class_counts": class_counts,
        "total_valid_samples": total_valid,
        "skipped_count": len(skipped_files),
        "stage1_features": STAGE1_FEATURE_NAMES,
        "stage2_features": STAGE2_FEATURE_NAMES,
        "stage1_info": stage1_info,
        "stage2_info": stage2_info,
    }

    save_trained_models(stage1_pipeline, stage2_pipeline, metadata)
    print("\n[4] Penyimpanan Model Selesai:")
    print(f"  • Model KNN Tahap 1 : {STAGE1_MODEL_PATH}")
    print(f"  • Model KNN Tahap 2 : {STAGE2_MODEL_PATH}")
    print(f"  • Metadata Model    : {METADATA_PATH}")
    print("=" * 76)
    return True


if __name__ == "__main__":
    run_training_pipeline()
