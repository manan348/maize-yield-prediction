import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.load_data import (
    get_genotype_info,
    load_phenotype_data,
    load_weather_data,
    print_dataset_info,
)
from src.data.preprocess import (
    prepare_phenotype_data,
    process_weather_data,
    merge_weather_with_phenotype,
)
from src.features.build_features import build_feature_matrix, load_snps_from_hdf5
from src.models.train import (
    cross_validate_model,
    evaluate_model,
    load_saved_model,
    save_model_artifacts,
    train_random_forest,
    train_test_data_split,
)


def run_lightweight_pipeline(output_dir: Path) -> None:
    """
    Fallback pipeline when genotype H5 is unavailable.

    Uses precomputed predictions and writes quick summary artifacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_predictions_path = output_dir / "all_predictions.csv"

    if not all_predictions_path.exists():
        raise FileNotFoundError(
            "Genotype file is missing and fallback file not found. "
            f"Expected: {all_predictions_path}"
        )

    print("Running lightweight pipeline from precomputed predictions...")
    df_preds = pd.read_csv(all_predictions_path)

    if "Yield" not in df_preds.columns:
        raise ValueError("all_predictions.csv must contain a 'Yield' column")

    # Global summary
    summary = {
        "rows": int(len(df_preds)),
        "yield_mean": float(df_preds["Yield"].mean()),
        "yield_std": float(df_preds["Yield"].std()),
        "yield_min": float(df_preds["Yield"].min()),
        "yield_max": float(df_preds["Yield"].max()),
    }
    (output_dir / "lightweight_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # Top crosses by location
    if {"Location", "Female", "Male", "Yield"}.issubset(df_preds.columns):
        top_crosses = (
            df_preds.sort_values(["Location", "Yield"], ascending=[True, False])
            .groupby("Location", as_index=False)
            .head(5)
            .reset_index(drop=True)
        )
        top_crosses.to_csv(output_dir / "top_crosses_by_location.csv", index=False)

    print("Lightweight pipeline completed successfully.")
    print(f"Saved summary: {output_dir / 'lightweight_metrics.json'}")


def run_predict_only_mode(output_dir: Path) -> None:
    """
    Run inference-only flow using saved artifacts.

    If saved artifacts are not available, fallback to precomputed summary mode.
    """
    model_path = output_dir / "model.joblib"
    x_final_path = output_dir / "X_final.npy"
    y_path = output_dir / "y.npy"

    if not (model_path.exists() and x_final_path.exists() and y_path.exists()):
        print(
            "Predict mode artifacts not found (model.joblib/X_final.npy/y.npy). "
            "Running lightweight fallback instead."
        )
        run_lightweight_pipeline(output_dir)
        return

    print("Running predict-only mode from saved model artifacts...")
    model = load_saved_model(str(model_path))
    X_final = np.load(str(x_final_path))
    y_true = np.load(str(y_path))

    y_pred = model.predict(X_final)
    pred_df = pd.DataFrame({
        "actual": y_true,
        "predicted": y_pred,
    })
    pred_df["error"] = (pred_df["actual"] - pred_df["predicted"]).abs()

    pred_csv = output_dir / "inference_predictions.csv"
    pred_df.to_csv(pred_csv, index=False)

    metrics = {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
    }
    (output_dir / "inference_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print("Predict-only mode completed successfully.")
    print(f"Saved predictions: {pred_csv}")
    print(f"Saved metrics: {output_dir / 'inference_metrics.json'}")


def run_pipeline(
    data_dir: Path,
    genotype_path: Path,
    output_dir: Path,
    use_fallback_if_missing_h5: bool = True,
) -> None:
    phe_path = data_dir / "g2f_2017_hybrid_data_clean.csv"
    wea_path = data_dir / "g2f_2017_weather_data.csv"

    if not genotype_path.exists():
        if use_fallback_if_missing_h5:
            print(
                "Genotype H5 not found; switching to lightweight mode using "
                "precomputed predictions."
            )
            run_lightweight_pipeline(output_dir)
            return
        raise FileNotFoundError(
            "Missing genotype file. Set --genotype-path to your .h5 file path. "
            f"Expected default: {genotype_path}"
        )

    if not phe_path.exists():
        raise FileNotFoundError(f"Missing phenotype file: {phe_path}")
    if not wea_path.exists():
        raise FileNotFoundError(f"Missing weather file: {wea_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting maize yield pipeline...")

    # 1) Load data
    df_phe = load_phenotype_data(str(phe_path), low_memory=False)
    df_wea = load_weather_data(str(wea_path), low_memory=False)
    n_taxa, n_snps = get_genotype_info(str(genotype_path))
    print_dataset_info(df_phe, df_wea, n_taxa, n_snps)

    # 2) Preprocess
    df, _, taxa_lookup = prepare_phenotype_data(df_phe, str(genotype_path))
    wea_season, wea_crit = process_weather_data(df_wea)
    df = merge_weather_with_phenotype(df, wea_season, wea_crit)

    # 3) Feature engineering
    X_female, X_male, y, X_env = load_snps_from_hdf5(
        str(genotype_path), df, taxa_lookup
    )
    feature_bundle = build_feature_matrix(X_female, X_male, X_env)

    # 4) Train model
    split_bundle = train_test_data_split(feature_bundle["X_final"], y)
    rf = train_random_forest(split_bundle["X_train_sc"], split_bundle["y_train"])
    eval_bundle = evaluate_model(
        rf,
        split_bundle["X_train_sc"],
        split_bundle["X_test_sc"],
        split_bundle["y_train"],
        split_bundle["y_test"],
    )
    cv_scores = cross_validate_model(feature_bundle["X_final"], y, cv=5)

    # 5) Save model artifacts
    save_model_artifacts(
        rf=rf,
        pca=feature_bundle["pca"],
        scaler_snp=feature_bundle["scaler_snp"],
        scaler_final=split_bundle["scaler"],
        X_snp=feature_bundle["X_snp"],
        X_env=X_env,
        X_final=feature_bundle["X_final"],
        y=y,
        df=df,
        save_path=str(output_dir) + "/",
    )

    # 6) Save predictions + metrics
    pred_df = pd.DataFrame(
        {
            "actual": split_bundle["y_test"],
            "predicted": eval_bundle["y_test_pred"],
        }
    )
    pred_df["error"] = (pred_df["actual"] - pred_df["predicted"]).abs()
    pred_csv = output_dir / "test_predictions.csv"
    pred_df.to_csv(pred_csv, index=False)

    metrics = {
        "train_r2": float(eval_bundle["train_r2"]),
        "test_r2": float(eval_bundle["test_r2"]),
        "cv_mean_r2": float(cv_scores.mean()),
        "cv_std_r2": float(cv_scores.std()),
        "mae": float(mean_absolute_error(split_bundle["y_test"], eval_bundle["y_test_pred"])),
        "rmse": float(mean_squared_error(split_bundle["y_test"], eval_bundle["y_test_pred"]) ** 0.5),
    }
    metrics_json = output_dir / "metrics.json"
    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Pipeline completed successfully.")
    print(f"Saved predictions: {pred_csv}")
    print(f"Saved metrics: {metrics_json}")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    default_data_dir = project_root / "data"
    default_genotype_path = default_data_dir / "g2f_2017_ZeaGBSv27_Imputed_AGPv4.h5"
    default_output_dir = project_root / "outputs" / "predictions"

    parser = argparse.ArgumentParser(description="Run maize yield ML pipeline.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir,
        help="Directory containing dataset CSV files.",
    )
    parser.add_argument(
        "--genotype-path",
        type=Path,
        default=default_genotype_path,
        help="Path to genotype H5 file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory to store model artifacts and predictions.",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "predict"],
        default="train",
        help="Run full train pipeline or inference-only mode.",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable fallback mode when genotype H5 is missing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "predict":
        run_predict_only_mode(args.output_dir)
    else:
        run_pipeline(
            args.data_dir,
            args.genotype_path,
            args.output_dir,
            use_fallback_if_missing_h5=not args.no_fallback,
        )
