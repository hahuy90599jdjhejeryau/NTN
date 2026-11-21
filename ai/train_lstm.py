from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

FEATURES = [
    "elevation_deg",
    "distance_km",
    "range_rate_km_s",
    "doppler_khz",
    "delay_ms",
    "fspl_db",
    "rx_power_dbm",
    "snr_db",
    "above_horizon",
]
TARGET = "snr_db"


class SNRLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.10 if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(x)
        return self.head(sequence[:, -1, :]).squeeze(-1)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_bool_column(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    mapped = series.astype(str).str.strip().str.lower().map(
        {"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0}
    )
    if mapped.isna().any():
        raise ValueError(f"Cột {series.name} không thể chuyển thành True/False.")
    return mapped.astype(np.float32)


def load_and_validate(csv_path: Path, minimum_coverage_fraction: float) -> tuple[pd.DataFrame, list[str], float]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy Dataset: {csv_path}")

    df = pd.read_csv(csv_path)
    required = ["scenario_id", "sat_id", "time_s", "candidate", *FEATURES]
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError("Dataset thiếu các cột bắt buộc: " + ", ".join(missing))
    if df.empty:
        raise ValueError("Dataset không có bản ghi.")

    df = df.copy()
    df["scenario_id"] = df["scenario_id"].astype(str)
    df["sat_id"] = df["sat_id"].astype(str)
    df["candidate"] = parse_bool_column(df["candidate"]).astype(bool)
    df["above_horizon"] = parse_bool_column(df["above_horizon"]).astype(np.float32)
    df[FEATURES[:-1] + ["time_s"]] = df[FEATURES[:-1] + ["time_s"]].apply(pd.to_numeric, errors="coerce")

    if df[required].isna().any().any():
        bad = df[required].columns[df[required].isna().any()].tolist()
        raise ValueError("Có giá trị trống/không phải số trong các cột: " + ", ".join(bad))
    if not np.isfinite(df[FEATURES[:-1] + ["time_s"]].to_numpy(dtype=np.float64)).all():
        raise ValueError("Dataset có giá trị NaN hoặc vô cực trong các trường số.")
    if df.duplicated(["scenario_id", "sat_id", "time_s"]).any():
        raise ValueError("Dataset có bản ghi trùng (scenario_id, sat_id, time_s).")

    scenario_ids = sorted(df["scenario_id"].unique().tolist())
    if len(scenario_ids) < 5:
        raise ValueError(
            f"Chỉ có {len(scenario_ids)} kịch bản. Cần ít nhất 5 kịch bản để chia train/validation/test theo scenario_id."
        )

    coverage = df.groupby("scenario_id")["candidate"].sum().astype(int)
    scenarios_with_candidates = int((coverage > 0).sum())
    required_covered = int(np.ceil(len(scenario_ids) * minimum_coverage_fraction))
    if scenarios_with_candidates < required_covered:
        detail = ", ".join(f"{sid}: {int(count)}" for sid, count in coverage.items())
        raise ValueError(
            "Dataset chưa phù hợp để huấn luyện AI lựa chọn liên kết: "
            f"chỉ {scenarios_with_candidates}/{len(scenario_ids)} kịch bản có candidate, "
            f"cần ít nhất {required_covered}. Số mẫu candidate theo kịch bản: {detail}. "
            "Hãy tạo lại CSV bằng phiên bản Step 2 đã sửa bộ chọn RAAN/pha."
        )

    time_steps: list[float] = []
    for (scenario_id, sat_id), group in df.groupby(["scenario_id", "sat_id"]):
        times = np.sort(group["time_s"].unique().astype(float))
        if len(times) < 40:
            raise ValueError(f"Chuỗi {scenario_id}/{sat_id} quá ngắn để tạo lookback 30 mẫu.")
        diffs = np.diff(times)
        if np.any(diffs <= 0) or not np.allclose(diffs, diffs[0]):
            raise ValueError(f"Chuỗi thời gian {scenario_id}/{sat_id} không đều hoặc không tăng liên tục.")
        time_steps.append(float(diffs[0]))
    if not np.allclose(time_steps, time_steps[0]):
        raise ValueError("Các chuỗi có chu kỳ lấy mẫu khác nhau. Hãy tạo Dataset với cùng chu kỳ lấy mẫu.")

    df = df.sort_values(["scenario_id", "sat_id", "time_s"]).reset_index(drop=True)
    return df, scenario_ids, time_steps[0]


def split_scenarios(scenario_ids: list[str], seed: int = 42) -> tuple[list[str], list[str], list[str]]:
    ids = scenario_ids.copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_test = max(1, int(round(n * 0.20)))
    n_val = max(1, int(round(n * 0.10)))
    while n - n_test - n_val < 1:
        if n_test > 1:
            n_test -= 1
        else:
            n_val -= 1
    test_ids = sorted(ids[:n_test])
    val_ids = sorted(ids[n_test:n_test + n_val])
    train_ids = sorted(ids[n_test + n_val:])
    return train_ids, val_ids, test_ids


def make_windows(
    df: pd.DataFrame,
    scenario_ids: list[str],
    x_scaler: StandardScaler | None,
    lookback: int,
    horizon_steps: int,
    window_stride: int,
    active_only: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    sequences: list[np.ndarray] = []
    targets: list[float] = []
    delta_targets: list[float] = []
    persistence: list[float] = []
    metadata: list[dict[str, Any]] = []

    subset = df[df["scenario_id"].isin(scenario_ids)]
    for (scenario_id, sat_id), group in subset.groupby(["scenario_id", "sat_id"], sort=True):
        group = group.sort_values("time_s")
        raw_x = group[FEATURES].to_numpy(dtype=np.float32)
        raw_y = group[TARGET].to_numpy(dtype=np.float32)
        above = group["above_horizon"].to_numpy(dtype=np.float32)
        times = group["time_s"].to_numpy(dtype=float)
        if x_scaler is not None:
            raw_x = x_scaler.transform(raw_x).astype(np.float32)

        final_start = len(group) - lookback - horizon_steps + 1
        for start in range(0, max(0, final_start), max(1, int(window_stride))):
            end = start + lookback
            target_idx = end - 1 + horizon_steps
            # Chỉ giữ các cửa sổ mà vệ tinh đang hoặc sắp trên đường chân trời (vùng handover thực tế)
            if active_only and (above[end - 1] < 0.5 and above[target_idx] < 0.5):
                continue

            cur_snr = float(raw_y[end - 1])
            fut_snr = float(raw_y[target_idx])
            sequences.append(raw_x[start:end])
            targets.append(fut_snr)
            delta_targets.append(fut_snr - cur_snr)
            persistence.append(cur_snr)
            metadata.append(
                {
                    "scenario_id": scenario_id,
                    "sat_id": sat_id,
                    "input_end_time_s": float(times[end - 1]),
                    "target_time_s": float(times[target_idx]),
                }
            )

    if not sequences:
        raise ValueError("Không tạo được cửa sổ thời gian. Kiểm tra lookback/horizon và độ dài Dataset.")
    return (
        np.stack(sequences).astype(np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(delta_targets, dtype=np.float32),
        np.asarray(persistence, dtype=np.float32),
        metadata,
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_pred - y_true
    return {
        "MAE_dB": float(np.mean(np.abs(error))),
        "RMSE_dB": float(np.sqrt(np.mean(error ** 2))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện LSTM dự báo SNR cho NTN LEO.")
    parser.add_argument("--csv", default="data/ntn_link_quality_dataset.csv", help="Đường dẫn Dataset CSV")
    parser.add_argument("--output-dir", default="results/step3_lstm", help="Thư mục lưu mô hình và kết quả")
    parser.add_argument("--lookback", type=int, default=30, help="Số mẫu quá khứ, mặc định 30")
    parser.add_argument("--horizon-seconds", type=float, default=5.0, help="Dự báo trước bao nhiêu giây")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--window-stride", type=int, default=1, help="Bước trượt cửa sổ (1 = dùng toàn bộ chuỗi)")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-coverage-fraction", type=float, default=0.70)
    args = parser.parse_args()

    set_seed(42)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    csv_path = Path(args.csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    model_dir = Path("models").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    df, scenario_ids, sample_interval = load_and_validate(csv_path, args.min_coverage_fraction)
    horizon_steps_float = args.horizon_seconds / sample_interval
    horizon_steps = int(round(horizon_steps_float))
    if horizon_steps < 1 or not np.isclose(horizon_steps_float, horizon_steps):
        raise ValueError(
            f"Horizon {args.horizon_seconds}s không chia hết cho chu kỳ lấy mẫu {sample_interval}s."
        )

    train_ids, val_ids, test_ids = split_scenarios(scenario_ids)
    print("\n=== PHÂN CHIA KỊCH BẢN (ACTIVE LINK WINDOWS) ===")
    print(f"Train ({len(train_ids)}): {train_ids}")
    print(f"Validation ({len(val_ids)}): {val_ids}")
    print(f"Test ({len(test_ids)}): {test_ids}")

    # Fit x_scaler trên vùng active của tập train để độ phân giải chuẩn hóa tốt nhất
    train_rows = df[(df["scenario_id"].isin(train_ids)) & (df["above_horizon"] > 0.5)]
    if len(train_rows) < 100:
        train_rows = df[df["scenario_id"].isin(train_ids)]
    x_scaler = StandardScaler()
    x_scaler.fit(train_rows[FEATURES].to_numpy(dtype=np.float64))

    X_train, y_train_raw, dy_train_raw, _, _ = make_windows(
        df, train_ids, x_scaler, args.lookback, horizon_steps, args.window_stride, active_only=True
    )
    X_val, y_val_raw, dy_val_raw, _, _ = make_windows(
        df, val_ids, x_scaler, args.lookback, horizon_steps, args.window_stride, active_only=True
    )
    X_test, y_test_raw, _, y_persist, test_meta = make_windows(
        df, test_ids, x_scaler, args.lookback, horizon_steps, args.window_stride, active_only=True
    )

    # Học độ biến thiên Delta-SNR (Residual Learning) để bắt chính xác quỹ đạo Doppler/FSPL
    y_scaler = StandardScaler()
    dy_train_scaled = y_scaler.fit_transform(dy_train_raw.reshape(-1, 1)).astype(np.float32).ravel()
    dy_val_scaled = y_scaler.transform(dy_val_raw.reshape(-1, 1)).astype(np.float32).ravel()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị: {device} | Active Windows: train={len(X_train):,}, val={len(X_val):,}, test={len(X_test):,}")

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(dy_train_scaled))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(dy_val_scaled))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = SNRLSTM(
        input_size=len(FEATURES),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = nn.SmoothL1Loss(beta=0.5)

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    wait = 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_batch_losses = []
        with torch.no_grad():
            for val_batch_x, val_batch_y in val_loader:
                val_batch_x = val_batch_x.to(device)
                val_batch_y = val_batch_y.to(device)
                val_prediction = model(val_batch_x)
                val_batch_losses.append(float(criterion(val_prediction, val_batch_y).cpu()))
        val_loss = float(np.mean(val_batch_losses))
        train_loss = float(np.mean(batch_losses))
        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping tại epoch {epoch}.")
                break

    if best_state is None:
        raise RuntimeError("Không lưu được trạng thái mô hình tốt nhất.")
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test)), batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_predictions_scaled: list[np.ndarray] = []
    with torch.no_grad():
        for (test_batch_x,) in test_loader:
            test_predictions_scaled.append(model(test_batch_x.to(device)).cpu().numpy())
    dy_test_scaled_pred = np.concatenate(test_predictions_scaled).reshape(-1, 1)
    dy_test_pred = y_scaler.inverse_transform(dy_test_scaled_pred).ravel()
    # Cộng độ biến thiên dự báo vào SNR hiện tại
    y_test_pred = y_persist + dy_test_pred

    lstm_metrics = regression_metrics(y_test_raw, y_test_pred)
    persistence_metrics = regression_metrics(y_test_raw, y_persist)

    metrics = {
        "target": "SNR after forecast horizon (Residual Delta-SNR on active links)",
        "predicts_delta": True,
        "lookback_samples": int(args.lookback),
        "sample_interval_seconds": float(sample_interval),
        "horizon_seconds": float(args.horizon_seconds),
        "window_stride": int(args.window_stride),
        "train_scenarios": train_ids,
        "validation_scenarios": val_ids,
        "test_scenarios": test_ids,
        "train_windows": int(len(X_train)),
        "validation_windows": int(len(X_val)),
        "test_windows": int(len(X_test)),
        "LSTM": lstm_metrics,
        "Persistence_baseline": persistence_metrics,
    }

    predictions = pd.DataFrame(test_meta)
    predictions["snr_actual_db"] = y_test_raw
    predictions["snr_lstm_pred_db"] = y_test_pred
    predictions["snr_persistence_pred_db"] = y_persist
    predictions["abs_error_lstm_db"] = np.abs(y_test_pred - y_test_raw)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"epoch": np.arange(1, len(train_losses) + 1), "train_loss": train_losses, "validation_loss": val_losses}).to_csv(
        output_dir / "training_history.csv", index=False
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "scenario_split.json").write_text(
        json.dumps({"train": train_ids, "validation": val_ids, "test": test_ids}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_features": FEATURES,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "predicts_delta": True,
            "lookback": args.lookback,
            "horizon_seconds": args.horizon_seconds,
            "window_stride": args.window_stride,
            "sample_interval_seconds": sample_interval,
        },
        model_dir / "lstm_snr.pth",
    )
    joblib.dump({"x_scaler": x_scaler, "y_scaler": y_scaler, "predicts_delta": True}, model_dir / "lstm_scalers.joblib")

    plt.figure(figsize=(9, 5))
    plt.plot(np.arange(1, len(train_losses) + 1), train_losses, label="Training loss")
    plt.plot(np.arange(1, len(val_losses) + 1), val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("SmoothL1 loss (scaled Delta-SNR)")
    plt.title("LSTM training history")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=180)
    plt.close()

    first_sequence = predictions[["scenario_id", "sat_id"]].drop_duplicates().iloc[0]
    view = predictions[
        (predictions["scenario_id"] == first_sequence["scenario_id"])
        & (predictions["sat_id"] == first_sequence["sat_id"])
    ].head(1500)
    plt.figure(figsize=(12, 5))
    plt.plot(view["target_time_s"], view["snr_actual_db"], label="Actual SNR", linewidth=1.5)
    plt.plot(view["target_time_s"], view["snr_lstm_pred_db"], label="LSTM prediction", linewidth=1.2)
    plt.plot(view["target_time_s"], view["snr_persistence_pred_db"], label="Persistence baseline", linewidth=1.0, alpha=0.8)
    plt.xlabel("Target time (s)")
    plt.ylabel("SNR (dB)")
    plt.title(f"Actual vs predicted SNR — {first_sequence['scenario_id']} / {first_sequence['sat_id']}")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "prediction_vs_actual.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(y_test_pred - y_test_raw, bins=40)
    plt.xlabel("Prediction error (dB)")
    plt.ylabel("Count")
    plt.title("LSTM test prediction error")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "error_histogram.png", dpi=180)
    plt.close()

    print("\n=== KẾT QUẢ TEST (ACTIVE LINKS) ===")
    print(f"LSTM       : MAE={lstm_metrics['MAE_dB']:.4f} dB | RMSE={lstm_metrics['RMSE_dB']:.4f} dB")
    print(f"Persistence: MAE={persistence_metrics['MAE_dB']:.4f} dB | RMSE={persistence_metrics['RMSE_dB']:.4f} dB")
    if lstm_metrics["MAE_dB"] < persistence_metrics["MAE_dB"]:
        improvement = (1.0 - lstm_metrics["MAE_dB"] / persistence_metrics["MAE_dB"]) * 100.0
        print(f"DAT CHUAN: LSTM giam {improvement:.1f}% sai so MAE so voi Persistence!")
    else:
        print("LSTM chưa vượt baseline. Cần điều chỉnh dữ liệu/siêu tham số; không kết luận AI tốt hơn.")
    print(f"\nĐã lưu mô hình: {model_dir / 'lstm_snr.pth'}")
    print(f"Đã lưu kết quả: {output_dir}")


if __name__ == "__main__":
    main()