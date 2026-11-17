from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from image_processing import (
    ALL_DATASET_CLASSES,
    ALLOWED_EXTENSIONS,
    BASE_DIR,
    RIPENESS_CLASS_CODES,
    STAGE1_MODEL_PATH,
    STAGE2_MODEL_PATH,
    SegmentationError,
    extract_both_stage_features_from_file,
    is_model_available,
    predict_two_stage_from_vectors,
)

TESTING_DATA_DIR = BASE_DIR / "dataset" / "testing"
RESULTS_DIR = BASE_DIR / "results"


def load_testing_dataset(dataset_dir: Path = TESTING_DATA_DIR) -> Dict[str, Any]:
    samples: List[Dict[str, Any]] = []
    class_counts: Dict[str, int] = {cls: 0 for cls in ALL_DATASET_CLASSES}
    skipped_files: List[Tuple[str, str]] = []

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Direktori dataset testing tidak ditemukan: {dataset_dir}"
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
                vec1, vec2, feats = extract_both_stage_features_from_file(img_path)
            except SegmentationError as seg_err:
                skipped_files.append((str(img_path.relative_to(BASE_DIR)), str(seg_err)))
                continue
            except Exception as exc:
                skipped_files.append((str(img_path.relative_to(BASE_DIR)), f"Error: {exc}"))
                continue

            class_counts[cls_name] += 1
            samples.append(
                {
                    "path": str(img_path.relative_to(BASE_DIR)),
                    "true_class": cls_name,
                    "true_binary": "bukan_pisang" if cls_name == "bukan_pisang" else "pisang",
                    "vec_stage1": vec1,
                    "vec_stage2": vec2,
                    "features": feats,
                }
            )

    return {
        "samples": samples,
        "class_counts": class_counts,
        "skipped_files": skipped_files,
    }


def compute_metrics_bundle(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
) -> Dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(
        precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )
    rec_macro = float(
        recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )
    f1_macro = float(
        f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )
    f1_weighted = float(
        f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report_str = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
        output_dict=True,
    )

    return {
        "accuracy": round(acc * 100.0, 2),
        "precision_macro": round(prec_macro * 100.0, 2),
        "recall_macro": round(rec_macro * 100.0, 2),
        "f1_macro": round(f1_macro * 100.0, 2),
        "f1_weighted": round(f1_weighted * 100.0, 2),
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "classification_report_text": report_str,
        "classification_report_dict": report_dict,
    }


def save_confusion_matrix_plot(
    cm: List[List[int]],
    labels: List[str],
    title: str,
    output_path: Path,
) -> None:
    cm_arr = np.array(cm, dtype=int)
    fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=150)
    im = ax.imshow(cm_arr, interpolation="nearest", cmap="Greens")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    display_labels = [lbl.replace("_", " ").title() for lbl in labels]
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=display_labels,
        yticklabels=display_labels,
        title=title,
        ylabel="Label Sebenarnya (Actual)",
        xlabel="Label Prediksi (Predicted)",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm_arr.max() / 2.0 if cm_arr.size > 0 else 0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(
                j,
                i,
                format(cm_arr[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm_arr[i, j] > thresh else "#1C2821",
                fontweight="bold",
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def run_evaluation_pipeline() -> bool:
    print("=" * 76)
    print("CAVENDISH VISION — EVALUASI KINERJA MODEL KNN DUA TAHAP")
    print("=" * 76)

    if not is_model_available():
        print("\n[PERINGATAN: MODEL BELUM DILATIH]")
        print("File model KNN belum ditemukan di dalam direktori `models/`:")
        print(f"  - {STAGE1_MODEL_PATH}")
        print(f"  - {STAGE2_MODEL_PATH}")
        print("Silakan jalankan `python train.py` terlebih dahulu setelah dataset training terisi.")
        print("Sistem TIDAK menampilkan angka evaluasi palsu.")
        print("=" * 76)
        return False

    test_data = load_testing_dataset(TESTING_DATA_DIR)
    samples: List[Dict[str, Any]] = test_data["samples"]
    class_counts: Dict[str, int] = test_data["class_counts"]

    print("\n[1] Jumlah Citra Testing Valid per Kelas:")
    print("-" * 76)
    for cls_name in ALL_DATASET_CLASSES:
        print(f"  • {cls_name:<18} : {class_counts[cls_name]} citra")
    print("-" * 76)
    print(f"  TOTAL CITRA UJI      : {len(samples)} citra")

    if len(samples) == 0:
        print("\n[PERINGATAN: DATASET TESTING MASIH KOSONG]")
        print(f"Tidak ditemukan citra uji pada direktori `{TESTING_DATA_DIR}`.")
        print("Silakan lengkapi dataset testing terlebih dahulu sebelum menjalankan evaluasi.")
        print("Sistem TIDAK menghasilkan metrik evaluasi palsu.")
        print("=" * 76)
        return False

    stage1_model = joblib.load(STAGE1_MODEL_PATH)
    stage2_model = joblib.load(STAGE2_MODEL_PATH)

    y_true_s1 = [s["true_binary"] for s in samples]
    x_s1 = np.array([s["vec_stage1"] for s in samples], dtype=np.float64)
    y_pred_s1 = [str(p) for p in stage1_model.predict(x_s1)]
    stage1_metrics = compute_metrics_bundle(
        y_true_s1,
        y_pred_s1,
        labels=["pisang", "bukan_pisang"],
    )

    banana_samples = [s for s in samples if s["true_class"] in RIPENESS_CLASS_CODES]
    stage2_metrics = None
    if banana_samples:
        y_true_s2 = [s["true_class"] for s in banana_samples]
        x_s2 = np.array([s["vec_stage2"] for s in banana_samples], dtype=np.float64)
        y_pred_s2 = [str(p) for p in stage2_model.predict(x_s2)]
        stage2_metrics = compute_metrics_bundle(
            y_true_s2,
            y_pred_s2,
            labels=RIPENESS_CLASS_CODES,
        )

    y_true_overall: List[str] = []
    y_pred_overall: List[str] = []
    for s in samples:
        y_true_overall.append(s["true_class"])
        pred_res = predict_two_stage_from_vectors(
            s["vec_stage1"],
            s["vec_stage2"],
            s["features"],
        )
        y_pred_overall.append(str(pred_res["predicted_code"]))

    overall_metrics = compute_metrics_bundle(
        y_true_overall,
        y_pred_overall,
        labels=ALL_DATASET_CLASSES,
    )

    print("\n[2] Hasil Evaluasi Model Tahap 1 (Pisang vs Bukan Pisang):")
    print(f"  • Accuracy        : {stage1_metrics['accuracy']:.2f}%")
    print(f"  • Precision Macro : {stage1_metrics['precision_macro']:.2f}%")
    print(f"  • Recall Macro    : {stage1_metrics['recall_macro']:.2f}%")
    print(f"  • F1-Score Macro  : {stage1_metrics['f1_macro']:.2f}%")
    print("\nClassification Report Tahap 1:\n" + stage1_metrics["classification_report_text"])

    if stage2_metrics is not None:
        print("[3] Hasil Evaluasi Model Tahap 2 (Klasifikasi 4 Tingkat Kematangan):")
        print(f"  • Accuracy        : {stage2_metrics['accuracy']:.2f}%")
        print(f"  • Precision Macro : {stage2_metrics['precision_macro']:.2f}%")
        print(f"  • Recall Macro    : {stage2_metrics['recall_macro']:.2f}%")
        print(f"  • F1-Score Macro  : {stage2_metrics['f1_macro']:.2f}%")
        print("\nClassification Report Tahap 2:\n" + stage2_metrics["classification_report_text"])

    print("[4] Hasil Evaluasi Pipeline Sistem Keseluruhan (End-to-End 5 Kelas):")
    print(f"  • Accuracy        : {overall_metrics['accuracy']:.2f}%")
    print(f"  • Precision Macro : {overall_metrics['precision_macro']:.2f}%")
    print(f"  • Recall Macro    : {overall_metrics['recall_macro']:.2f}%")
    print(f"  • F1-Score Macro  : {overall_metrics['f1_macro']:.2f}%")
    print("\nClassification Report Keseluruhan:\n" + overall_metrics["classification_report_text"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix_plot(
        stage1_metrics["confusion_matrix"],
        stage1_metrics["labels"],
        "Confusion Matrix Tahap 1 (Pisang vs Bukan Pisang)",
        RESULTS_DIR / "confusion_matrix_stage1.png",
    )
    if stage2_metrics is not None:
        save_confusion_matrix_plot(
            stage2_metrics["confusion_matrix"],
            stage2_metrics["labels"],
            "Confusion Matrix Tahap 2 (Tingkat Kematangan)",
            RESULTS_DIR / "confusion_matrix_stage2.png",
        )
    save_confusion_matrix_plot(
        overall_metrics["confusion_matrix"],
        overall_metrics["labels"],
        "Confusion Matrix Pipeline Keseluruhan (5 Kelas)",
        RESULTS_DIR / "confusion_matrix_overall.png",
    )

    full_report = {
        "evaluated_at": datetime.now().isoformat(),
        "total_test_samples": len(samples),
        "test_class_counts": class_counts,
        "stage1_binary_evaluation": stage1_metrics,
        "stage2_ripeness_evaluation": stage2_metrics,
        "overall_pipeline_evaluation": overall_metrics,
    }

    json_report_path = RESULTS_DIR / "evaluation_report.json"
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    txt_report_path = RESULTS_DIR / "classification_report.txt"
    with open(txt_report_path, "w", encoding="utf-8") as f:
        f.write("LAPORAN EVALUASI SISTEM CAVENDISH VISION\n")
        f.write("=" * 72 + "\n\n")
        f.write("1. MODEL TAHAP 1 (IDENTIFIKASI PISANG VS BUKAN PISANG)\n")
        f.write(stage1_metrics["classification_report_text"] + "\n")
        if stage2_metrics is not None:
            f.write("2. MODEL TAHAP 2 (KLASIFIKASI 4 TINGKAT KEMATANGAN)\n")
            f.write(stage2_metrics["classification_report_text"] + "\n")
        f.write("3. PIPELINE SISTEM KESELURUHAN (END-TO-END)\n")
        f.write(overall_metrics["classification_report_text"] + "\n")

    print("\n[5] Seluruh Hasil Evaluasi Disimpan ke Folder `results/`:")
    print(f"  • {json_report_path}")
    print(f"  • {txt_report_path}")
    print(f"  • {RESULTS_DIR / 'confusion_matrix_stage1.png'}")
    if stage2_metrics is not None:
        print(f"  • {RESULTS_DIR / 'confusion_matrix_stage2.png'}")
    print(f"  • {RESULTS_DIR / 'confusion_matrix_overall.png'}")
    print("=" * 76)
    return True


if __name__ == "__main__":
    run_evaluation_pipeline()
