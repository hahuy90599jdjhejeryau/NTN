from __future__ import annotations

import json
import math
import urllib.request
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image
from torch import nn

# Small LSTM inference workloads are often slower when BLAS/PyTorch spawns
# dozens of CPU threads. Keep a modest thread count; numerical results do not change.
try:
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
except RuntimeError:
    pass

# =============================================================================
# NTN LEO - STEP 1 TO 6: CONSTELLATION / ORBIT / VISIBILITY / AI BEAM HANDOVER
# Don vi: km, s, Hz, dB, dBm, Mbps.
# =============================================================================
R_EARTH_KM = 6371.0
MU_KM3_S2 = 398600.4418
OMEGA_EARTH_RAD_S = 7.2921159e-5
C_KM_S = 299792.458
MIN_ELEVATION_DEG = 10.0

TX_POWER_DBM = 30.0
TX_GAIN_KA_DBI = 30.0
RX_GAIN_KA_DBI = 40.0
SYSTEM_LOSS_DB = 2.0
ATM_LOSS_KA_DB = 3.0
FREQ_KA_GHZ = 28.0
BW_KA_HZ = 100e6
NOISE_FIGURE_DB = 3.0
RX_SENSITIVITY_DBM = -100.0

# Cau hinh cum 7 Spot Beams tren moi ve tinh (1 Beam trung tam + 6 Beam xung quanh)
NUM_BEAMS_PER_SAT = 7
BEAM_HALF_POWER_ANGLE_DEG = 18.0  # Goc nua cong suat bup song (-3 dB)

GROUND_STATIONS: dict[str, tuple[float, float]] = {
    "TĐ. Hà Nội": (21.0285, 105.8542),
    "TP. Đà Nẵng": (16.0544, 108.2022),
    "TP.HCM": (10.8231, 106.6297),
    "QĐ. Hoàng Sa": (16.5333, 111.6167),
    "QĐ. Trường Sa": (8.6436, 111.9197),
}

TEXTURE_URLS = [
    "https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/planets/earth_atmos_2048.jpg",
    "https://www.inf.u-szeged.hu/~tanacs/threejs/examples/textures/planets/earth_atmos_2048.jpg",
]

SETTINGS_PATH = Path(__file__).resolve().parent / "settings.json"
DEFAULT_SETTINGS = {"station_name":"TĐ. Hà Nội","sim_time":0,"altitude_km":600,"inclination_deg":53,"n_planes":3,"sats_per_plane":4,"raan_center":98,"plane_spacing":4,"phase_start":24,"ho_duration":720,"hyst_margin":2.0,"ttt_sec":3,"ai_weight":0.65}
def load_settings() -> dict[str, Any]:
    try: return {**DEFAULT_SETTINGS, **json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))}
    except Exception: return DEFAULT_SETTINGS.copy()
def save_settings(settings: dict[str, Any]) -> None:
    SETTINGS_PATH.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")

LSTM_FEATURES = [
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


class SNRLSTMInference(nn.Module):
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


def get_lstm_artifact_signature() -> tuple[int, int, int, int]:
    """Version key for cache invalidation when model/scaler files are replaced."""
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "models" / "lstm_snr.pth"
    scaler_path = base_dir / "models" / "lstm_scalers.joblib"
    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            "Chưa tìm thấy models/lstm_snr.pth hoặc models/lstm_scalers.joblib. Hãy huấn luyện LSTM trước."
        )
    model_stat = model_path.stat()
    scaler_stat = scaler_path.stat()
    return (
        int(model_stat.st_mtime_ns), int(model_stat.st_size),
        int(scaler_stat.st_mtime_ns), int(scaler_stat.st_size),
    )


@st.cache_resource(show_spinner=False)
def load_trained_lstm_bundle(
    artifact_signature: tuple[int, int, int, int],
) -> tuple[SNRLSTMInference, dict[str, Any], dict[str, Any]]:
    """Load model once; signature ensures replacing model files invalidates the cache."""
    del artifact_signature  # Used by Streamlit as part of the resource-cache key.
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "models" / "lstm_snr.pth"
    scaler_path = base_dir / "models" / "lstm_scalers.joblib"
    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            "Chưa tìm thấy models/lstm_snr.pth hoặc models/lstm_scalers.joblib. Hãy huấn luyện LSTM trước."
        )
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    scalers = joblib.load(scaler_path)
    features = ckpt.get("input_features", LSTM_FEATURES)
    hidden_size = int(ckpt.get("hidden_size", 64))
    num_layers = int(ckpt.get("num_layers", 2))
    model = SNRLSTMInference(input_size=len(features), hidden_size=hidden_size, num_layers=num_layers)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, scalers, ckpt


def latlon_to_ecef(lat_deg: float, lon_deg: float, radius_km: float = R_EARTH_KM) -> np.ndarray:
    """Chuyen vi do/kinh do sang toa do ECEF, cung quy uoc voi file goc."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    return np.array(
        [
            radius_km * math.cos(lat) * math.cos(lon),
            radius_km * math.cos(lat) * math.sin(lon),
            radius_km * math.sin(lat),
        ],
        dtype=float,
    )


def orbit_state_ecef(
    altitude_km: float,
    inclination_deg: float,
    raan_deg: float,
    theta_rad: float,
    elapsed_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vi tri va van toc ve tinh ECEF tu mo hinh quy dao tron Kepler ECI."""
    r_sat = R_EARTH_KM + altitude_km
    inc = math.radians(inclination_deg)
    raan = math.radians(raan_deg)
    mean_motion = math.sqrt(MU_KM3_S2 / (r_sat**3))

    xp = r_sat * math.cos(theta_rad)
    yp = r_sat * math.sin(theta_rad)
    vxp = -r_sat * mean_motion * math.sin(theta_rad)
    vyp = r_sat * mean_motion * math.cos(theta_rad)

    cO, sO = math.cos(raan), math.sin(raan)
    ci, si = math.cos(inc), math.sin(inc)

    pos_eci = np.array(
        [xp * cO - yp * ci * sO, xp * sO + yp * ci * cO, yp * si], dtype=float
    )
    vel_eci = np.array(
        [vxp * cO - vyp * ci * sO, vxp * sO + vyp * ci * cO, vyp * si], dtype=float
    )

    earth_angle = OMEGA_EARTH_RAD_S * elapsed_s
    ca, sa = math.cos(earth_angle), math.sin(earth_angle)

    def rotate_eci_to_ecef(vec: np.ndarray) -> np.ndarray:
        return np.array(
            [ca * vec[0] + sa * vec[1], -sa * vec[0] + ca * vec[1], vec[2]],
            dtype=float,
        )

    pos_ecef = rotate_eci_to_ecef(pos_eci)
    vel_rotated = rotate_eci_to_ecef(vel_eci)
    vel_ecef = vel_rotated - np.cross(
        np.array([0.0, 0.0, OMEGA_EARTH_RAD_S]), pos_ecef
    )
    return pos_ecef, vel_ecef


def evaluate_satellite_beams(
    pos_sat_ecef: np.ndarray,
    vel_sat_ecef: np.ndarray,
    pos_gs_ecef: np.ndarray,
    base_snr_db: float,
) -> dict[str, Any]:
    """
    Mo hinh 7 Spot Beams tren moi ve tinh (1 Beam trung tam + 6 Beam xung quanh).
    Tinh goc lech giua huong UE voi tam tung Beam de chon Beam toi uu (Beam_best).
    """
    rho = np.asarray(pos_gs_ecef, dtype=float) - np.asarray(pos_sat_ecef, dtype=float)
    dist = float(np.linalg.norm(rho))
    los_unit = rho / max(dist, 1e-9)

    # Truc z_sat huong ve tam Trai Dat (Nadir)
    z_sat = -np.asarray(pos_sat_ecef, dtype=float)
    z_sat /= max(float(np.linalg.norm(z_sat)), 1e-9)

    # Truc x_sat doc theo phuong chuyen dong ngang
    v_tan = np.asarray(vel_sat_ecef, dtype=float) - np.dot(vel_sat_ecef, z_sat) * z_sat
    if float(np.linalg.norm(v_tan)) < 1e-9:
        v_tan = np.array([1.0, 0.0, 0.0])
    x_sat = v_tan / float(np.linalg.norm(v_tan))
    y_sat = np.cross(z_sat, x_sat)

    # Dinh nghia 7 huong tam Beam (1 Nadir + 6 vien luc giac goc nghieng 24 deg)
    beam_specs: list[tuple[str, float, float]] = [("B01-Center", 0.0, 0.0)]
    ring_tilt_rad = math.radians(24.0)
    for idx in range(6):
        az_rad = math.radians(idx * 60.0)
        beam_specs.append((f"B0{idx + 2}-Spot", ring_tilt_rad, az_rad))

    best_beam_id = "B01-Center"
    best_beam_gain_offset_db = -999.0
    best_off_axis_deg = 0.0
    all_beams: list[dict[str, Any]] = []

    for b_id, tilt, az in beam_specs:
        b_dir = (
            math.cos(tilt) * z_sat
            + math.sin(tilt) * math.cos(az) * x_sat
            + math.sin(tilt) * math.sin(az) * y_sat
        )
        b_dir /= max(float(np.linalg.norm(b_dir)), 1e-9)
        cos_psi = float(np.clip(np.dot(los_unit, b_dir), -1.0, 1.0))
        off_axis_deg = math.degrees(math.acos(cos_psi))

        # Mo hinh suy hao bup song Gaussian (-3 dB tai goc BEAM_HALF_POWER_ANGLE_DEG)
        gain_offset_db = -3.0 * ((off_axis_deg / BEAM_HALF_POWER_ANGLE_DEG) ** 2)
        gain_offset_db = max(gain_offset_db, -25.0)
        beam_snr_db = base_snr_db + gain_offset_db

        all_beams.append(
            {
                "beam_id": b_id,
                "off_axis_deg": round(off_axis_deg, 2),
                "beam_gain_offset_db": round(gain_offset_db, 2),
                "beam_snr_db": round(beam_snr_db, 2),
            }
        )
        if gain_offset_db > best_beam_gain_offset_db:
            best_beam_gain_offset_db = gain_offset_db
            best_beam_id = b_id
            best_off_axis_deg = off_axis_deg

    return {
        "best_beam_id": best_beam_id,
        "best_beam_offset_db": float(best_beam_gain_offset_db),
        "best_off_axis_deg": float(best_off_axis_deg),
        "beam_snr_db": float(base_snr_db + best_beam_gain_offset_db),
        "beams": all_beams,
    }


def link_metrics(
    pos_sat_ecef: np.ndarray,
    vel_sat_ecef: np.ndarray,
    pos_gs_ecef: np.ndarray,
) -> dict[str, Any]:
    """Tinh slant range, Elevation, Doppler, Link Budget Ka-band va chon Beam toi uu."""
    rho = np.asarray(pos_sat_ecef, dtype=float) - np.asarray(pos_gs_ecef, dtype=float)
    distance_km = float(np.linalg.norm(rho))
    if not math.isfinite(distance_km) or distance_km <= 0:
        raise ValueError("Khoang cach ve tinh - tram mat dat khong hop le.")

    up = pos_gs_ecef / np.linalg.norm(pos_gs_ecef)
    sin_el = float(np.dot(rho, up) / distance_km)
    elevation_deg = math.degrees(math.asin(float(np.clip(sin_el, -1.0, 1.0))))

    range_rate_km_s = float(np.dot(vel_sat_ecef, rho / distance_km))
    doppler_khz = -(FREQ_KA_GHZ * 1e9) * (range_rate_km_s / C_KM_S) / 1000.0
    delay_ms = (distance_km / C_KM_S) * 1000.0

    frequency_mhz = FREQ_KA_GHZ * 1000.0
    fspl_db = 32.44 + 20.0 * math.log10(distance_km) + 20.0 * math.log10(frequency_mhz)
    blockage_loss_db = 50.0 if elevation_deg <= 0.0 else 0.0
    rx_power_dbm = (
        TX_POWER_DBM + TX_GAIN_KA_DBI + RX_GAIN_KA_DBI
        - fspl_db - SYSTEM_LOSS_DB - ATM_LOSS_KA_DB - blockage_loss_db
    )
    noise_dbm = -174.0 + 10.0 * math.log10(BW_KA_HZ) + NOISE_FIGURE_DB
    snr_db = rx_power_dbm - noise_dbm

    beam_info = evaluate_satellite_beams(pos_sat_ecef, vel_sat_ecef, pos_gs_ecef, snr_db)
    above_horizon = elevation_deg > 0.0
    candidate = elevation_deg >= MIN_ELEVATION_DEG and rx_power_dbm >= RX_SENSITIVITY_DBM

    return {
        "distance_km": distance_km,
        "elevation_deg": elevation_deg,
        "range_rate_km_s": range_rate_km_s,
        "doppler_khz": doppler_khz,
        "delay_ms": delay_ms,
        "fspl_db": fspl_db,
        "rx_power_dbm": rx_power_dbm,
        "snr_db": snr_db,
        "above_horizon": above_horizon,
        "candidate": candidate,
        "best_beam_id": beam_info["best_beam_id"],
        "best_beam_offset_db": beam_info["best_beam_offset_db"],
        "best_off_axis_deg": beam_info["best_off_axis_deg"],
        "beam_snr_db": beam_info["beam_snr_db"],
        "beams": beam_info["beams"],
    }


def make_constellation(
    n_planes: int,
    sats_per_plane: int,
    altitude_km: float,
    inclination_deg: float,
    raan_center_deg: float,
    plane_spacing_deg: float,
    phase_start_deg: float,
) -> list[dict[str, Any]]:
    """Tao danh sach day du cac ve tinh; khong loc theo Visibility tai day."""
    satellites: list[dict[str, Any]] = []
    for plane_idx in range(n_planes):
        raan = (raan_center_deg + (plane_idx - (n_planes - 1) / 2.0) * plane_spacing_deg) % 360.0
        for sat_idx in range(sats_per_plane):
            phase = (phase_start_deg + sat_idx * 360.0 / sats_per_plane) % 360.0
            satellites.append(
                {
                    "sat_id": f"LEO-P{plane_idx + 1:02d}-S{sat_idx + 1:02d}",
                    "plane": plane_idx + 1,
                    "raan_deg": raan,
                    "phase_deg": phase,
                    "altitude_km": altitude_km,
                    "inclination_deg": inclination_deg,
                }
            )
    return satellites


def get_satellite_states(
    satellites: list[dict[str, Any]], elapsed_s: float, pos_gs: np.ndarray
) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for sat in satellites:
        r_sat = R_EARTH_KM + float(sat["altitude_km"])
        mean_motion = math.sqrt(MU_KM3_S2 / r_sat**3)
        theta = math.radians(float(sat["phase_deg"])) + mean_motion * elapsed_s
        pos, vel = orbit_state_ecef(
            float(sat["altitude_km"]),
            float(sat["inclination_deg"]),
            float(sat["raan_deg"]),
            theta,
            elapsed_s,
        )
        metrics = link_metrics(pos, vel, pos_gs)
        states.append({**sat, "pos": pos, "vel": vel, **metrics})
    return states


def _fast_scenario_coverage_score(
    raan_center_deg: float, phase_start_deg: float, n_planes: int,
    sats_per_plane: int, altitude_km: float, inclination_deg: float,
    plane_spacing_deg: float, time_values: np.ndarray, pos_gs: np.ndarray,
) -> tuple[int, int, float]:
    """Ước lượng số mẫu liên kết đạt ngưỡng để chọn pha/RAAN trước khi sinh CSV."""
    times = np.asarray(time_values, dtype=float)
    radius = R_EARTH_KM + float(altitude_km)
    n = math.sqrt(MU_KM3_S2 / radius**3)
    inc = math.radians(float(inclination_deg))
    ci, si = math.cos(inc), math.sin(inc)
    earth_angle = OMEGA_EARTH_RAD_S * times
    ca, sa = np.cos(earth_angle), np.sin(earth_angle)
    up = pos_gs / np.linalg.norm(pos_gs)
    candidate_count = 0
    above_count = 0
    max_elevation = -90.0
    freq_mhz = FREQ_KA_GHZ * 1000.0

    for plane_idx in range(int(n_planes)):
        raan = (float(raan_center_deg) + (plane_idx - (n_planes - 1) / 2.0) * float(plane_spacing_deg)) % 360.0
        O = math.radians(raan)
        cO, sO = math.cos(O), math.sin(O)
        for sat_idx in range(int(sats_per_plane)):
            phase = (float(phase_start_deg) + sat_idx * 360.0 / sats_per_plane) % 360.0
            theta = math.radians(phase) + n * times
            xp, yp = radius * np.cos(theta), radius * np.sin(theta)
            x_eci = xp * cO - yp * ci * sO
            y_eci = xp * sO + yp * ci * cO
            z_eci = yp * si
            x = ca * x_eci + sa * y_eci
            y = -sa * x_eci + ca * y_eci
            z = z_eci
            dx, dy, dz = x - pos_gs[0], y - pos_gs[1], z - pos_gs[2]
            distance = np.sqrt(dx * dx + dy * dy + dz * dz)
            sin_el = (dx * up[0] + dy * up[1] + dz * up[2]) / np.maximum(distance, 1e-12)
            elevation = np.degrees(np.arcsin(np.clip(sin_el, -1.0, 1.0)))
            fspl = 32.44 + 20.0 * np.log10(np.maximum(distance, 1e-9)) + 20.0 * math.log10(freq_mhz)
            blockage = np.where(elevation <= 0.0, 50.0, 0.0)
            rx_power = TX_POWER_DBM + TX_GAIN_KA_DBI + RX_GAIN_KA_DBI - fspl - SYSTEM_LOSS_DB - ATM_LOSS_KA_DB - blockage
            candidate = (elevation >= MIN_ELEVATION_DEG) & (rx_power >= RX_SENSITIVITY_DBM)
            candidate_count += int(np.count_nonzero(candidate))
            above_count += int(np.count_nonzero(elevation > 0.0))
            max_elevation = max(max_elevation, float(np.max(elevation)))
    return candidate_count, above_count, max_elevation


@st.cache_data(show_spinner=False, max_entries=24)
def build_coverage_scenario_configs(
    scenario_count: int, raan_center_deg: float, phase_start_deg: float,
    n_planes: int, sats_per_plane: int, altitude_km: float,
    inclination_deg: float, plane_spacing_deg: float,
    time_values: np.ndarray, pos_gs: np.ndarray,
) -> list[dict[str, float | int]]:
    """Chọn cấu hình RAAN/pha có pass qua UE."""
    phases = sorted({round((float(phase_start_deg) + offset) % 90.0, 3) for offset in np.arange(0.0, 90.0, 3.0)})
    scored: list[dict[str, float | int]] = []
    for span in (16.0, 40.0):
        for offset in np.arange(-span, span + 0.001, 4.0):
            raan = (float(raan_center_deg) + float(offset)) % 360.0
            for phase in phases:
                cand, above, max_el = _fast_scenario_coverage_score(
                    raan, phase, n_planes, sats_per_plane, altitude_km,
                    inclination_deg, plane_spacing_deg, time_values, pos_gs,
                )
                if cand > 0:
                    scored.append({
                        "raan_center_deg": raan,
                        "phase_start_deg": phase,
                        "candidate_samples": cand,
                        "above_horizon_samples": above,
                        "max_elevation_deg": max_el,
                    })
        if len(scored) >= int(scenario_count):
            break
    if not scored:
        raise ValueError(
            "Không tìm được kịch bản có vệ tinh đạt Elevation tối thiểu. "
            "Hãy tăng thời gian mô phỏng hoặc kiểm tra độ cao, góc nghiêng và RAAN."
        )
    scored.sort(key=lambda x: (int(x["candidate_samples"]), float(x["max_elevation_deg"])), reverse=True)

    def circ_dist(a: float, b: float, period: float) -> float:
        diff = abs(a - b) % period
        return min(diff, period - diff)

    chosen: list[dict[str, float | int]] = []
    for item in scored:
        diverse = True
        for old in chosen:
            dr = circ_dist(float(item["raan_center_deg"]), float(old["raan_center_deg"]), 360.0) / 16.0
            dp = circ_dist(float(item["phase_start_deg"]), float(old["phase_start_deg"]), 90.0) / 45.0
            if math.sqrt(dr * dr + dp * dp) < 0.32:
                diverse = False
                break
        if diverse:
            chosen.append(item)
        if len(chosen) >= int(scenario_count):
            break

    if len(chosen) < int(scenario_count):
        used = {(round(float(x["raan_center_deg"]), 3), round(float(x["phase_start_deg"]), 3)) for x in chosen}
        for item in scored:
            key = (round(float(item["raan_center_deg"]), 3), round(float(item["phase_start_deg"]), 3))
            if key not in used:
                chosen.append(item)
                used.add(key)
            if len(chosen) >= int(scenario_count):
                break
    if len(chosen) < int(scenario_count):
        raise ValueError(f"Chỉ tìm được {len(chosen)} kịch bản có liên kết; hãy tăng thời gian mô phỏng hoặc nới cấu hình quỹ đạo.")
    return chosen[:int(scenario_count)]


# =============================================================================
# STEP 4-6: AI SATELLITE & BEAM SELECTION + HANDOVER KPI COMPARISON ENGINE
# =============================================================================
@st.cache_data(show_spinner=False, max_entries=24)
def simulate_constellation_timeseries(
    n_planes: int,
    sats_per_plane: int,
    altitude_km: float,
    inclination_deg: float,
    raan_center_deg: float,
    plane_spacing_deg: float,
    phase_start_deg: float,
    station_name: str,
    duration_s: int = 720,
    sample_interval_s: int = 1,
) -> pd.DataFrame:
    """Vectorized Kepler/ECI-ECEF/link-budget/7-beam simulation.

    The formulas and thresholds match get_satellite_states()/link_metrics(),
    but calculations are performed as NumPy arrays over all times and satellites
    rather than nested Python loops. Results retain the same columns and units.
    """
    n_planes = int(n_planes)
    sats_per_plane = int(sats_per_plane)
    duration_s = int(duration_s)
    sample_interval_s = int(sample_interval_s)
    if n_planes < 1 or sats_per_plane < 1 or duration_s < 0 or sample_interval_s < 1:
        raise ValueError("Cấu hình chòm sao, thời lượng hoặc chu kỳ lấy mẫu không hợp lệ.")

    station_lat, station_lon = GROUND_STATIONS[station_name]
    pos_gs = latlon_to_ecef(station_lat, station_lon)
    up = pos_gs / np.linalg.norm(pos_gs)

    # Keep the same satellite ordering/IDs as make_constellation().
    plane_idx = np.repeat(np.arange(n_planes, dtype=np.int32), sats_per_plane)
    sat_idx = np.tile(np.arange(sats_per_plane, dtype=np.int32), n_planes)
    sat_count = n_planes * sats_per_plane
    raan_deg = (float(raan_center_deg) + (plane_idx - (n_planes - 1) / 2.0) * float(plane_spacing_deg)) % 360.0
    phase_deg = (float(phase_start_deg) + sat_idx * 360.0 / sats_per_plane) % 360.0
    sat_ids = np.array([
        f"LEO-P{p + 1:02d}-S{s + 1:02d}"
        for p in range(n_planes) for s in range(sats_per_plane)
    ], dtype=object)

    radius = R_EARTH_KM + float(altitude_km)
    mean_motion = math.sqrt(MU_KM3_S2 / radius**3)
    inc = math.radians(float(inclination_deg))
    ci, si = math.cos(inc), math.sin(inc)
    raan = np.radians(raan_deg)
    cO, sO = np.cos(raan), np.sin(raan)
    time_values = np.arange(0, duration_s + 1, sample_interval_s, dtype=np.int32)
    time_axis = time_values.astype(np.float64)
    theta = np.radians(phase_deg)[None, :] + mean_motion * time_axis[:, None]
    earth_angle = OMEGA_EARTH_RAD_S * time_axis
    ca, sa = np.cos(earth_angle)[:, None], np.sin(earth_angle)[:, None]

    # Orbital-plane position/velocity -> ECI.
    xp = radius * np.cos(theta)
    yp = radius * np.sin(theta)
    vxp = -radius * mean_motion * np.sin(theta)
    vyp = radius * mean_motion * np.cos(theta)
    x_eci = xp * cO[None, :] - yp * ci * sO[None, :]
    y_eci = xp * sO[None, :] + yp * ci * cO[None, :]
    z_eci = yp * si
    vx_eci = vxp * cO[None, :] - vyp * ci * sO[None, :]
    vy_eci = vxp * sO[None, :] + vyp * ci * cO[None, :]
    vz_eci = vyp * si

    # ECI -> ECEF. v_ECEF = R(v_ECI) - Omega x r_ECEF.
    x = ca * x_eci + sa * y_eci
    y = -sa * x_eci + ca * y_eci
    z = z_eci
    vx_rot = ca * vx_eci + sa * vy_eci
    vy_rot = -sa * vx_eci + ca * vy_eci
    vz_rot = vz_eci
    vx = vx_rot + OMEGA_EARTH_RAD_S * y
    vy = vy_rot - OMEGA_EARTH_RAD_S * x
    vz = vz_rot

    # Link geometry and Ka-band budget.
    dx, dy, dz = x - pos_gs[0], y - pos_gs[1], z - pos_gs[2]
    distance = np.sqrt(dx * dx + dy * dy + dz * dz)
    inv_distance = 1.0 / np.maximum(distance, 1e-12)
    sin_el = (dx * up[0] + dy * up[1] + dz * up[2]) * inv_distance
    elevation = np.degrees(np.arcsin(np.clip(sin_el, -1.0, 1.0)))
    range_rate = (vx * dx + vy * dy + vz * dz) * inv_distance
    doppler = -(FREQ_KA_GHZ * 1e9) * (range_rate / C_KM_S) / 1000.0
    delay = (distance / C_KM_S) * 1000.0
    frequency_mhz = FREQ_KA_GHZ * 1000.0
    fspl = 32.44 + 20.0 * np.log10(np.maximum(distance, 1e-9)) + 20.0 * math.log10(frequency_mhz)
    blockage = np.where(elevation <= 0.0, 50.0, 0.0)
    rx_power = TX_POWER_DBM + TX_GAIN_KA_DBI + RX_GAIN_KA_DBI - fspl - SYSTEM_LOSS_DB - ATM_LOSS_KA_DB - blockage
    noise_dbm = -174.0 + 10.0 * math.log10(BW_KA_HZ) + NOISE_FIGURE_DB
    snr = rx_power - noise_dbm
    above = elevation > 0.0
    candidate = (elevation >= MIN_ELEVATION_DEG) & (rx_power >= RX_SENSITIVITY_DBM)

    # Vectorized equivalent of evaluate_satellite_beams(): 1 nadir beam + 6 ring beams.
    pos = np.stack((x, y, z), axis=-1)                 # (time, satellite, xyz)
    vel = np.stack((vx, vy, vz), axis=-1)
    los = np.stack((pos_gs[0] - x, pos_gs[1] - y, pos_gs[2] - z), axis=-1) * inv_distance[..., None]
    z_sat = -pos / np.maximum(np.linalg.norm(pos, axis=-1, keepdims=True), 1e-12)
    v_radial = np.sum(vel * z_sat, axis=-1, keepdims=True)
    v_tan = vel - v_radial * z_sat
    x_sat = v_tan / np.maximum(np.linalg.norm(v_tan, axis=-1, keepdims=True), 1e-12)
    y_sat = np.cross(z_sat, x_sat)
    tilts = np.r_[0.0, np.full(6, math.radians(24.0))]
    azimuths = np.r_[0.0, np.radians(np.arange(6) * 60.0)]
    beam_dirs = (
        np.cos(tilts)[None, None, :, None] * z_sat[:, :, None, :]
        + (np.sin(tilts) * np.cos(azimuths))[None, None, :, None] * x_sat[:, :, None, :]
        + (np.sin(tilts) * np.sin(azimuths))[None, None, :, None] * y_sat[:, :, None, :]
    )
    cos_off_axis = np.sum(los[:, :, None, :] * beam_dirs, axis=-1)
    off_axis = np.degrees(np.arccos(np.clip(cos_off_axis, -1.0, 1.0)))
    beam_offsets = np.maximum(-3.0 * (off_axis / BEAM_HALF_POWER_ANGLE_DEG) ** 2, -25.0)
    best_beam_index = np.argmax(beam_offsets, axis=2)
    best_offset = np.take_along_axis(beam_offsets, best_beam_index[:, :, None], axis=2)[:, :, 0]
    best_off_axis = np.take_along_axis(off_axis, best_beam_index[:, :, None], axis=2)[:, :, 0]
    beam_names = np.array(["B01-Center"] + [f"B0{i + 2}-Spot" for i in range(6)], dtype=object)
    best_beam_id = beam_names[best_beam_index]
    beam_snr = snr + best_offset

    # Preserve the existing time-major row order and DataFrame schema.
    time_col = np.repeat(time_values, sat_count)
    sat_col = np.tile(sat_ids, len(time_values))
    plane_col = np.tile(plane_idx + 1, len(time_values))
    result = pd.DataFrame({
        "time_s": time_col,
        "sat_id": sat_col,
        "plane": plane_col,
        "elevation_deg": elevation.reshape(-1),
        "distance_km": distance.reshape(-1),
        "range_rate_km_s": range_rate.reshape(-1),
        "doppler_khz": doppler.reshape(-1),
        "delay_ms": delay.reshape(-1),
        "fspl_db": fspl.reshape(-1),
        "rx_power_dbm": rx_power.reshape(-1),
        "snr_db": snr.reshape(-1),
        "above_horizon": above.astype(np.float32).reshape(-1),
        "candidate": candidate.reshape(-1),
        "best_beam_id": best_beam_id.reshape(-1),
        "best_beam_offset_db": best_offset.reshape(-1),
        "best_off_axis_deg": best_off_axis.reshape(-1),
        "beam_snr_db": beam_snr.reshape(-1),
    })
    return result


@st.cache_data(show_spinner=False, max_entries=16)
def cached_lstm_predictions(
    _ts_df: pd.DataFrame,
    data_signature: tuple[Any, ...],
    artifact_signature: tuple[int, int, int, int],
    inference_batch_size: int = 4096,
) -> pd.DataFrame:
    """Cache LSTM forecasts and run one bounded batch stream over sliding windows.

    _ts_df is intentionally excluded from Streamlit hashing; data_signature is
    the complete physical simulation configuration and artifact_signature
    invalidates the forecast cache when the trained model/scalers are replaced.
    """
    del data_signature  # Included in Streamlit cache key, not needed by inference.
    model, scalers, ckpt = load_trained_lstm_bundle(artifact_signature)
    features = list(ckpt.get("input_features", LSTM_FEATURES))
    lookback = int(ckpt.get("lookback", 30))
    predicts_delta = bool(ckpt.get("predicts_delta", scalers.get("predicts_delta", False)))
    x_scaler = scalers["x_scaler"]
    y_scaler = scalers["y_scaler"]
    if lookback < 1:
        raise ValueError(f"Lookback không hợp lệ trong checkpoint: {lookback}")
    missing = [name for name in features if name not in _ts_df.columns]
    required = {"snr_db", "above_horizon", "best_beam_offset_db"}
    missing.extend(name for name in required if name not in _ts_df.columns and name not in missing)
    if missing:
        raise ValueError("Thiếu cột dữ liệu cho LSTM/Beam: " + ", ".join(missing))

    groups: list[pd.DataFrame] = []
    pred_arrays: list[np.ndarray] = []
    window_blocks: list[np.ndarray] = []
    window_targets: list[tuple[int, np.ndarray]] = []
    offsets = np.arange(lookback, dtype=np.int64)

    # One independent temporal sequence per satellite; never bridge sat_id boundaries.
    for group_idx, (_sat_id, source_group) in enumerate(_ts_df.groupby("sat_id", sort=True)):
        group = source_group.sort_values("time_s", kind="stable").reset_index(drop=True).copy()
        raw_x = group[features].to_numpy(dtype=np.float32, copy=True)
        scaled_x = x_scaler.transform(raw_x).astype(np.float32, copy=False)
        current_snr = group["snr_db"].to_numpy(dtype=np.float32, copy=True)
        above = group["above_horizon"].to_numpy(dtype=np.float32, copy=False)
        groups.append(group)
        pred_arrays.append(current_snr)
        if len(group) < lookback:
            continue

        target_idx = np.flatnonzero(above[lookback - 1:] > 0.5) + lookback - 1
        if target_idx.size == 0:
            continue
        starts = target_idx - lookback + 1
        window_index = starts[:, None] + offsets[None, :]
        # Shape: (number of valid predictions, lookback, feature_count).
        windows = np.ascontiguousarray(scaled_x[window_index], dtype=np.float32)
        window_blocks.append(windows)
        window_targets.append((group_idx, target_idx))

    total_windows = sum(block.shape[0] for block in window_blocks)
    if total_windows:
        all_windows = np.concatenate(window_blocks, axis=0)
        batch_size = max(64, int(inference_batch_size))
        raw_outputs: list[np.ndarray] = []
        model.eval()
        with torch.inference_mode():
            for start in range(0, total_windows, batch_size):
                batch = torch.from_numpy(all_windows[start:start + batch_size])
                raw_outputs.append(model(batch).detach().cpu().numpy().reshape(-1, 1))
        scaled_outputs = np.concatenate(raw_outputs, axis=0)
        predictions = y_scaler.inverse_transform(scaled_outputs).ravel().astype(np.float32)

        cursor = 0
        for group_idx, target_idx in window_targets:
            count = target_idx.size
            values = predictions[cursor:cursor + count]
            if predicts_delta:
                pred_arrays[group_idx][target_idx] = pred_arrays[group_idx][target_idx] + values
            else:
                pred_arrays[group_idx][target_idx] = values
            cursor += count
        del all_windows, raw_outputs, scaled_outputs, predictions

    output_parts: list[pd.DataFrame] = []
    for group, predicted_snr in zip(groups, pred_arrays):
        group["pred_snr_5s_db"] = predicted_snr
        group["pred_beam_snr_5s_db"] = (
            group["pred_snr_5s_db"].astype(float)
            + group["best_beam_offset_db"].astype(float)
        )
        output_parts.append(group)

    result = pd.concat(output_parts, ignore_index=True)
    return result.sort_values(["time_s", "sat_id"], kind="stable").reset_index(drop=True)


def compute_shannon_throughput_mbps(effective_snr_db: float, is_handover_step: bool = False) -> float:
    """Tinh thong luong Shannon (Mbps) tren bang thong 100 MHz Ka-band; tru 15% overhead tai giay handover."""
    snr_linear = 10.0 ** (effective_snr_db / 10.0)
    cap_mbps = (BW_KA_HZ / 1e6) * math.log2(1.0 + max(snr_linear, 0.0))
    if is_handover_step:
        cap_mbps *= 0.85
    return float(cap_mbps)


def run_handover_policy(
    ts_df: pd.DataFrame,
    policy: str,
    hyst_margin_db: float = 2.0,
    ttt_seconds: int = 3,
    ai_weight: float = 0.65,
    pingpong_window_s: int = 15,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Low-overhead per-time-step policy engine using NumPy arrays.

    Decision rules, hysteresis/TTT conditions and KPI definitions are retained;
    DataFrame-wide filtering, repeated sorting and temporary score columns are removed.
    """
    if ts_df.empty:
        raise ValueError("Không có dữ liệu thời gian để chạy thuật toán Handover.")
    required = {
        "time_s", "sat_id", "candidate", "elevation_deg", "beam_snr_db", "snr_db",
        "pred_snr_5s_db", "pred_beam_snr_5s_db", "best_beam_id",
    }
    missing = sorted(required.difference(ts_df.columns))
    if missing:
        raise ValueError("Thiếu cột cho Handover: " + ", ".join(missing))

    ordered = ts_df.sort_values(["time_s", "sat_id"], kind="stable")
    serving_sat: str | None = None
    serving_beam: str | None = None
    last_ho_time: int | None = None
    prev_sat: str | None = None
    pending_target: str | None = None
    pending_count = 0
    sat_ho_count = 0
    beam_switch_count = 0
    pingpong_count = 0
    outage_steps = 0
    records: list[dict[str, Any]] = []
    hyst_margin_db = float(hyst_margin_db)
    ttt_seconds = max(1, int(ttt_seconds))
    ai_weight = float(np.clip(ai_weight, 0.0, 1.0))

    for t, frame in ordered.groupby("time_s", sort=False):
        sat_ids = frame["sat_id"].to_numpy(dtype=object)
        candidate = frame["candidate"].to_numpy(dtype=bool)
        elevation = frame["elevation_deg"].to_numpy(dtype=np.float64)
        beam_snr = frame["beam_snr_db"].to_numpy(dtype=np.float64)
        snr = frame["snr_db"].to_numpy(dtype=np.float64)
        pred_snr = frame["pred_snr_5s_db"].to_numpy(dtype=np.float64)
        pred_beam_snr = frame["pred_beam_snr_5s_db"].to_numpy(dtype=np.float64)
        beam_ids = frame["best_beam_id"].to_numpy(dtype=object)
        candidate_idx = np.flatnonzero(candidate)

        if candidate_idx.size == 0:
            serving_sat = None
            serving_beam = None
            pending_target = None
            pending_count = 0
            outage_steps += 1
            best_idx = int(np.argmax(snr))
            records.append({
                "time_s": int(t), "policy": policy, "serving_sat": "OUTAGE",
                "serving_beam": "—", "sat_beam_pair": "OUTAGE",
                "serving_snr_db": float(snr[best_idx]),
                "beam_snr_db": float(beam_snr[best_idx]),
                "pred_snr_5s_db": float(pred_snr[best_idx]),
                "serving_elev_deg": float(elevation[best_idx]),
                "throughput_mbps": 0.0, "is_candidate": False,
                "handover": False, "beam_switch": False, "pingpong": False,
            })
            continue

        if policy == "Rule-based (Max Elevation)":
            scores = elevation[candidate_idx]
        elif policy == "Rule-based + Hysteresis (TTT)":
            scores = beam_snr[candidate_idx]
        else:
            delta_pred = pred_snr[candidate_idx] - snr[candidate_idx]
            falling_penalty = np.where(delta_pred < -0.10, 1.5 * ai_weight * np.abs(delta_pred), 0.0)
            scores = (
                (1.0 - ai_weight) * beam_snr[candidate_idx]
                + ai_weight * pred_beam_snr[candidate_idx]
                + 0.8 * ai_weight * delta_pred
                - falling_penalty
            )

        best_local_idx = int(np.argmax(scores))
        best_idx = int(candidate_idx[best_local_idx])
        best_id = str(sat_ids[best_idx])
        best_score = float(scores[best_local_idx])
        current_matches = candidate_idx[sat_ids[candidate_idx] == serving_sat] if serving_sat is not None else np.empty(0, dtype=int)
        cur_idx = int(current_matches[0]) if current_matches.size else None
        handover_happened = False
        is_pingpong = False

        if cur_idx is None:
            if serving_sat is not None and serving_sat != best_id:
                handover_happened = True
            serving_sat = best_id
            pending_target = None
            pending_count = 0
        else:
            if policy == "Rule-based (Max Elevation)":
                cur_score = float(elevation[cur_idx])
            elif policy == "Rule-based + Hysteresis (TTT)":
                cur_score = float(beam_snr[cur_idx])
            else:
                cur_delta_for_score = float(pred_snr[cur_idx] - snr[cur_idx])
                cur_penalty = 1.5 * ai_weight * abs(cur_delta_for_score) if cur_delta_for_score < -0.10 else 0.0
                cur_score = (
                    (1.0 - ai_weight) * float(beam_snr[cur_idx])
                    + ai_weight * float(pred_beam_snr[cur_idx])
                    + 0.8 * ai_weight * cur_delta_for_score
                    - cur_penalty
                )

            if best_id == serving_sat:
                pending_target = None
                pending_count = 0
            else:
                if policy == "Rule-based (Max Elevation)":
                    serving_sat = best_id
                    handover_happened = True
                elif policy == "Rule-based + Hysteresis (TTT)":
                    if best_score >= cur_score + hyst_margin_db:
                        if pending_target == best_id:
                            pending_count += 1
                        else:
                            pending_target, pending_count = best_id, 1
                        if pending_count >= ttt_seconds:
                            serving_sat = best_id
                            handover_happened = True
                            pending_target, pending_count = None, 0
                    else:
                        pending_target, pending_count = None, 0
                else:
                    cur_delta = float(pred_snr[cur_idx] - snr[cur_idx])
                    cur_falling = (ai_weight > 0.0) and (cur_delta < -0.05)
                    effective_margin = hyst_margin_db * max(0.15, 1.0 - 0.85 * ai_weight) if cur_falling else hyst_margin_db
                    required_ttt = 1 if (cur_falling and ai_weight >= 0.5) else ttt_seconds
                    if best_score >= cur_score + effective_margin:
                        if pending_target == best_id:
                            pending_count += 1
                        else:
                            pending_target, pending_count = best_id, 1
                        if pending_count >= required_ttt:
                            serving_sat = best_id
                            handover_happened = True
                            pending_target, pending_count = None, 0
                    else:
                        pending_target, pending_count = None, 0

        chosen_matches = np.flatnonzero(sat_ids == serving_sat)
        if chosen_matches.size == 0:
            raise RuntimeError(f"Không tìm thấy bản ghi vệ tinh đang phục vụ {serving_sat} tại t={t}.")
        chosen_idx = int(chosen_matches[0])
        new_beam = str(beam_ids[chosen_idx])
        beam_switched = False
        if not handover_happened and serving_beam is not None and new_beam != serving_beam:
            beam_switched = True
            beam_switch_count += 1
        serving_beam = new_beam

        if handover_happened:
            sat_ho_count += 1
            if prev_sat is not None and serving_sat == prev_sat and last_ho_time is not None and (int(t) - last_ho_time) <= pingpong_window_s:
                pingpong_count += 1
                is_pingpong = True
            prev_sat = records[-1]["serving_sat"] if records else None
            last_ho_time = int(t)

        effective_beam_snr = float(beam_snr[chosen_idx])
        throughput = compute_shannon_throughput_mbps(effective_beam_snr, is_handover_step=handover_happened)
        records.append({
            "time_s": int(t), "policy": policy, "serving_sat": serving_sat,
            "serving_beam": serving_beam,
            "sat_beam_pair": f"{serving_sat} / {serving_beam}",
            "serving_snr_db": float(snr[chosen_idx]),
            "beam_snr_db": effective_beam_snr,
            "pred_snr_5s_db": float(pred_snr[chosen_idx]),
            "serving_elev_deg": float(elevation[chosen_idx]),
            "throughput_mbps": throughput, "is_candidate": True,
            "handover": handover_happened, "beam_switch": beam_switched,
            "pingpong": is_pingpong,
        })

    result_df = pd.DataFrame.from_records(records)
    active = result_df.loc[result_df["is_candidate"], "beam_snr_db"]
    mean_snr = float(active.mean()) if not active.empty else float("nan")
    mean_throughput = float(result_df["throughput_mbps"].mean())
    availability = float(result_df["is_candidate"].mean() * 100.0)
    pingpong_rate = float(pingpong_count / sat_ho_count * 100.0) if sat_ho_count else 0.0
    summary = {
        "Chiến lược": policy,
        "Satellite Handover": int(sat_ho_count),
        "Beam Switch": int(beam_switch_count),
        "Ping-Pong (%)": round(pingpong_rate, 2),
        "SNR sau Beam (dB)": round(mean_snr, 3) if math.isfinite(mean_snr) else None,
        "Throughput TB (Mbps)": round(mean_throughput, 2),
        "Link Availability (%)": round(availability, 2),
        "Outage Ratio (%)": round(100.0 - availability, 2),
        "Outage (s)": int(outage_steps),
    }
    return result_df, summary


@st.cache_data(show_spinner=False, max_entries=96)
def cached_handover_policy(
    _ts_df: pd.DataFrame,
    data_signature: tuple[Any, ...],
    artifact_signature: tuple[int, int, int, int],
    policy: str,
    hyst_margin_db: float,
    ttt_seconds: int,
    ai_weight: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Cache policy output by simulation/model signature and policy parameters."""
    del data_signature, artifact_signature
    return run_handover_policy(
        _ts_df,
        policy=policy,
        hyst_margin_db=float(hyst_margin_db),
        ttt_seconds=int(ttt_seconds),
        ai_weight=float(ai_weight),
    )


@st.cache_data(show_spinner=False, max_entries=24)
def cached_alpha_sensitivity(
    _pred_ts: pd.DataFrame,
    data_signature: tuple[Any, ...],
    artifact_signature: tuple[int, int, int, int],
    hyst_margin_db: float,
    ttt_seconds: int,
) -> pd.DataFrame:
    """Cached fixed alpha sweep; independent of the live AI-weight slider."""
    del data_signature, artifact_signature
    rows: list[dict[str, Any]] = []
    for weight in (0.0, 0.65, 1.0):
        _, summary = run_handover_policy(
            _pred_ts,
            policy=f"AI LSTM (α = {weight:.2f})",
            hyst_margin_db=float(hyst_margin_db),
            ttt_seconds=int(ttt_seconds),
            ai_weight=float(weight),
        )
        rows.append(summary)
    return pd.DataFrame(rows)


def locate_earth_texture() -> tuple[Path, str]:
    """Tim texture goc, thu tai texture neu chua co, cuoi cung dung ban do du phong."""
    base_dir = Path(__file__).resolve().parent
    local_candidates = [
        base_dir / "earth_texture.jpg",
        base_dir / "earth_atmos_2048.jpg",
    ]
    for path in local_candidates:
        if path.exists() and path.stat().st_size > 10_000:
            return path, path.name

    for url in TEXTURE_URLS:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 NTN-LEO-Simulator"})
            with urllib.request.urlopen(request, timeout=12) as response:
                image_bytes = response.read()
            target = base_dir / "earth_texture.jpg"
            target.write_bytes(image_bytes)
            if target.stat().st_size > 10_000:
                return target, "earth_texture.jpg (tai tu Three.js)"
        except Exception:
            continue

    fallback = base_dir / "bmng_preview.jpg"
    if fallback.exists():
        return fallback, "bmng_preview.jpg (anh du phong)"
    raise FileNotFoundError(
        "Khong tim thay texture Trai Dat. Hay dat earth_texture.jpg hoac bmng_preview.jpg "
        "cung thu muc voi app.py."
    )


@st.cache_data(show_spinner=False)
def earth_vertex_data(n_lat: int = 361, n_lon: int = 721) -> tuple[np.ndarray, ...]:
    """Luoi dia ly chi tiet cao va mau RGB lay truc tiep tu texture ban do."""
    texture_path, _texture_label = locate_earth_texture()
    with Image.open(texture_path) as source_image:
        image_np = np.asarray(source_image.convert("RGB"))
    img_h, img_w, _ = image_np.shape

    lon_deg = np.linspace(-180.0, 180.0, n_lon)
    lat_deg = np.linspace(-90.0, 90.0, n_lat)
    lon_grid, lat_grid = np.meshgrid(lon_deg, lat_deg)
    lon_rad = np.radians(lon_grid)
    lat_rad = np.radians(lat_grid)

    x = R_EARTH_KM * np.cos(lat_rad) * np.cos(lon_rad)
    y = R_EARTH_KM * np.cos(lat_rad) * np.sin(lon_rad)
    z = R_EARTH_KM * np.sin(lat_rad)

    col = np.clip(np.rint((lon_grid + 180.0) / 360.0 * (img_w - 1)).astype(int), 0, img_w - 1)
    row = np.clip(np.rint((90.0 - lat_grid) / 180.0 * (img_h - 1)).astype(int), 0, img_h - 1)
    rgb = image_np[row, col].reshape(-1, 3)
    vertex_colors = [f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in rgb]

    grid_ids = np.arange(n_lat * n_lon, dtype=np.int32).reshape(n_lat, n_lon)
    a = grid_ids[:-1, :-1].ravel()
    b = grid_ids[:-1, 1:].ravel()
    c = grid_ids[1:, 1:].ravel()
    d = grid_ids[1:, :-1].ravel()
    tri_i = np.concatenate([a, a])
    tri_j = np.concatenate([b, c])
    tri_k = np.concatenate([c, d])
    return x.ravel(), y.ravel(), z.ravel(), tri_i, tri_j, tri_k, np.asarray(vertex_colors, dtype=object)


def add_earth_and_grid(fig: go.Figure) -> None:
    """Dung qua cau ban do chi tiet, giu he kinh/vi do cua PyVista goc."""
    x, y, z, tri_i, tri_j, tri_k, vertex_colors = earth_vertex_data()
    fig.add_trace(
        go.Mesh3d(
            x=x, y=y, z=z,
            i=tri_i, j=tri_j, k=tri_k,
            vertexcolor=vertex_colors,
            flatshading=False,
            lighting=dict(ambient=0.96, diffuse=0.28, specular=0.02, roughness=1.0, fresnel=0.0),
            lightposition=dict(x=-15000, y=18000, z=14000),
            hoverinfo="skip",
            name="Trái Đất — bản đồ vệ tinh",
            showscale=False,
        )
    )

    for lon_value in range(-180, 181, 30):
        lat_line = np.linspace(-89.5, 89.5, 100)
        lon_r = math.radians(lon_value)
        lat_r = np.radians(lat_line)
        radius = R_EARTH_KM * 1.0017
        gx = radius * np.cos(lat_r) * math.cos(lon_r)
        gy = radius * np.cos(lat_r) * math.sin(lon_r)
        gz = radius * np.sin(lat_r)
        fig.add_trace(
            go.Scatter3d(
                x=gx, y=gy, z=gz, mode="lines",
                line=dict(color="rgba(220,239,255,0.20)", width=1),
                hoverinfo="skip", showlegend=False,
            )
        )
    for lat_value in range(-60, 61, 30):
        lon_line = np.linspace(-180.0, 180.0, 200)
        lat_r = math.radians(lat_value)
        lon_r = np.radians(lon_line)
        radius = R_EARTH_KM * 1.0017
        gx = radius * np.cos(lat_r) * np.cos(lon_r)
        gy = radius * np.cos(lat_r) * np.sin(lon_r)
        gz = np.full_like(gx, radius * math.sin(lat_r))
        fig.add_trace(
            go.Scatter3d(
                x=gx, y=gy, z=gz, mode="lines",
                line=dict(color="rgba(220,239,255,0.20)", width=1),
                hoverinfo="skip", showlegend=False,
            )
        )


def make_figure(
    satellites: list[dict[str, Any]],
    states: list[dict[str, Any]],
    selected_station: str,
    selected_station_pos: np.ndarray,
    elapsed_s: float,
    station_coords: dict[str, tuple[float, float]],
) -> go.Figure:
    fig = go.Figure()
    add_earth_and_grid(fig)

    planes: dict[int, dict[str, Any]] = {}
    for sat in satellites:
        planes.setdefault(int(sat["plane"]), sat)
    orbit_palette = ["#EAF1FA", "#1688E8", "#78D0FF", "#F0A34A", "#57C785", "#B18AE8"]
    theta_grid = np.linspace(0.0, 2.0 * np.pi, 240)
    for plane_idx, representative in planes.items():
        orbit_points = np.array(
            [
                orbit_state_ecef(
                    float(representative["altitude_km"]),
                    float(representative["inclination_deg"]),
                    float(representative["raan_deg"]),
                    float(theta),
                    elapsed_s,
                )[0]
                for theta in theta_grid
            ]
        )
        fig.add_trace(
            go.Scatter3d(
                x=orbit_points[:, 0], y=orbit_points[:, 1], z=orbit_points[:, 2],
                mode="lines",
                line=dict(color=orbit_palette[(plane_idx - 1) % len(orbit_palette)], width=4),
                name=f"Quỹ đạo mặt phẳng {plane_idx}",
                hoverinfo="skip",
            )
        )

    below_horizon = [s for s in states if not bool(s["above_horizon"])]
    above_but_not_ready = [s for s in states if bool(s["above_horizon"]) and not bool(s["candidate"])]
    ready = [s for s in states if bool(s["candidate"])]

    def add_satellite_group(items: list[dict[str, Any]], label: str, color: str, size: int, opacity: float) -> None:
        if not items:
            return
        xyz = np.array([s["pos"] for s in items])
        hover = [
            f"<b>{s['sat_id']}</b> (Beam tối ưu: {s['best_beam_id']})<br>Độ cao: {s['altitude_km']:.0f} km"
            f"<br>Elevation: {s['elevation_deg']:.2f}°"
            f"<br>Khoảng cách: {s['distance_km']:,.2f} km"
            f"<br>SNR kênh: {s['snr_db']:.2f} dB · SNR sau Beam: {s['beam_snr_db']:.2f} dB"
            f"<br>Trạng thái: {'ĐỦ ĐIỀU KIỆN' if s['candidate'] else ('TRÊN CHÂN TRỜI' if s['above_horizon'] else 'DƯỚI CHÂN TRỜI')}"
            for s in items
        ]
        fig.add_trace(
            go.Scatter3d(
                x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                mode="markers",
                marker=dict(size=size, color=color, opacity=opacity, line=dict(color="#F5FAFF", width=0.9)),
                text=hover, hovertemplate="%{text}<extra></extra>", name=label,
            )
        )

    add_satellite_group(ready, "Vệ tinh đủ điều kiện", "#35E39A", 8, 1.0)
    add_satellite_group(above_but_not_ready, "Trên chân trời — chưa đạt ngưỡng", "#FFB84D", 6, 0.95)
    add_satellite_group(below_horizon, "Dưới đường chân trời UE", "#BEC8D4", 5, 0.72)

    if ready:
        best = max(ready, key=lambda s: float(s["elevation_deg"]))
        for state in ready:
            pos = state["pos"]
            is_best = state["sat_id"] == best["sat_id"]
            fig.add_trace(
                go.Scatter3d(
                    x=[pos[0], selected_station_pos[0]],
                    y=[pos[1], selected_station_pos[1]],
                    z=[pos[2], selected_station_pos[2]],
                    mode="lines",
                    line=dict(color="#FFD166" if is_best else "rgba(54,227,154,0.60)", width=5 if is_best else 2),
                    hoverinfo="skip", showlegend=False,
                )
            )

    gs_xyz = np.array(
        [latlon_to_ecef(*coords, R_EARTH_KM * 1.004) for coords in station_coords.values()]
    )
    gs_names = list(station_coords.keys())
    gs_colors = ["#FFD166" if name == selected_station else "#FF7474" for name in gs_names]
    fig.add_trace(
        go.Scatter3d(
            x=gs_xyz[:, 0], y=gs_xyz[:, 1], z=gs_xyz[:, 2],
            mode="markers+text",
            marker=dict(size=[8 if n == selected_station else 6 for n in gs_names], color=gs_colors, line=dict(color="white", width=1)),
            text=gs_names, textposition="top center", textfont=dict(size=10, color="#FFFFFF"),
            name="Trạm mặt đất Việt Nam", hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[selected_station_pos[0]], y=[selected_station_pos[1]], z=[selected_station_pos[2]],
            mode="markers",
            marker=dict(size=11, color="#FFD166", symbol="diamond", line=dict(color="#FFFFFF", width=1.4)),
            name=f"UE / {selected_station}",
            hovertemplate=f"UE / {selected_station}<extra></extra>",
        )
    )

    total_satellites = len(satellites)
    ready_count = len(ready)
    geometric_count = sum(bool(s["above_horizon"]) for s in states)
    fig.update_layout(
        height=820,
        margin=dict(l=0, r=0, t=58, b=8),
        paper_bgcolor="#081321",
        plot_bgcolor="#081321",
        font=dict(family="Be Vietnam Pro, Arial, sans-serif", color="#F2F7FC", size=12),
        title=dict(
            text=(
                f"TRÁI ĐẤT 3D · CHÒM SAO LEO — t = {elapsed_s:.0f} s"
                f"<br><sup>{total_satellites} vệ tinh trong chòm sao · {geometric_count} trên chân trời · {ready_count} đủ điều kiện liên kết</sup>"
            ),
            x=0.025, y=0.985,
            font=dict(size=17, color="#F4F8FF"),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=0.015, xanchor="left", x=0.015,
            bgcolor="rgba(7,18,32,0.92)",
            bordercolor="rgba(153,190,220,0.45)", borderwidth=1,
            font=dict(family="Be Vietnam Pro, Arial, sans-serif", size=11, color="#F0F6FF"),
            itemclick="toggle", itemdoubleclick="toggleothers",
        ),
        scene=dict(
            bgcolor="#081321",
            xaxis=dict(visible=False, range=[-8200, 8200]),
            yaxis=dict(visible=False, range=[-8200, 8200]),
            zaxis=dict(visible=False, range=[-8200, 8200]),
            aspectmode="manual", aspectratio=dict(x=1, y=1, z=1),
            camera=dict(eye=dict(x=-0.83, y=2.56, z=0.77), up=dict(x=0, y=0, z=1)),
        ),
        uirevision="ntn-earth-camera",
        showlegend=True,
    )
    return fig


# =============================================================================
# WEB UI
# =============================================================================
st.set_page_config(
    page_title="NTN LEO | Satellite & Beam AI Handover Lab",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&display=swap');
    :root { --bg:#07111f; --panel:#11263b; --line:rgba(157,190,221,.22); --text:#f2f7ff; --muted:#b9cadd; --cyan:#69d9ef; --green:#43dda0; --gold:#ffd166; }
    html, body, [class*="css"], .stApp { font-family:'Be Vietnam Pro',Arial,sans-serif; }
    .stApp { background:radial-gradient(ellipse at 52% -18%,#1b4262 0%,#0b192a 43%,#060e19 100%); color:var(--text); }
    [data-testid="stHeader"] { background:rgba(6,14,25,.94); }
    [data-testid="stMainBlockContainer"] { padding-top:1.0rem; padding-bottom:2.4rem; max-width:2000px; }
    [data-testid="stSidebar"] { background:linear-gradient(180deg,#122a42 0%,#091725 100%); border-right:1px solid var(--line); }
    [data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,[data-testid="stSidebar"] label,[data-testid="stSidebar"] span { color:#F0F6FF !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] > div { background:#172f48 !important; border:1px solid #47627c !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] input,
    [data-testid="stSidebar"] [data-baseweb="select"] div,
    [data-testid="stSidebar"] [data-baseweb="select"] span { color:#F4F8FF !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] svg { fill:#EAF3FC !important; }
    [data-baseweb="popover"] [role="listbox"], [data-baseweb="popover"] [role="option"] { background:#142B42 !important; color:#F4F8FF !important; }
    [data-baseweb="popover"] [role="option"]:hover { background:#28506F !important; }
    .stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp p,.stApp label,.stApp li { color:#F0F6FF; }
    h1 { font-weight:800 !important; letter-spacing:-.5px; }
    [data-testid="stMetric"] { background:linear-gradient(145deg,#193752,#10243A); border:1px solid var(--line); border-radius:15px; padding:13px 15px; min-height:96px; }
    [data-testid="stMetricLabel"] { color:#C7D8E8 !important; font-size:.82rem !important; }
    [data-testid="stMetricValue"] { color:#FFFFFF !important; font-size:clamp(1.05rem,1.35vw,1.42rem) !important; font-weight:800 !important; white-space:normal !important; overflow:visible !important; text-overflow:clip !important; overflow-wrap:anywhere !important; line-height:1.25 !important; }
    [data-testid="stMetricDelta"] { color:#A9C1D7 !important; }
    [data-testid="stPlotlyChart"] { background:#081321; border:1px solid rgba(157,190,221,.22); border-radius:18px; padding:4px; box-shadow:0 14px 34px rgba(0,0,0,.28); }
    [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:13px; overflow:hidden; }
    [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span, [data-testid="stSlider"] label { color:#FFFFFF !important; font-weight:700 !important; text-shadow:0 1px 2px rgba(0,0,0,.5); }
    [data-testid="stMarkdown"] p, [data-testid="stCaption"] { color:#E1ECF7 !important; }
    [data-testid="stPlotlyChart"] svg text { fill:#F4F8FF !important; }
    [data-testid="stFormSubmitButton"] button { background:#087FA5 !important; color:#FFFFFF !important; border:1px solid #65D8F2 !important; font-size:1rem !important; font-weight:800 !important; }
    [data-testid="stFormSubmitButton"] button:hover { background:#0B9BC5 !important; }

    .hero { padding:22px 26px; margin:0 0 17px; border-radius:19px; background:linear-gradient(110deg,#153a58,#102a43 62%,#123d49); border:1px solid rgba(90,190,220,.34); box-shadow:0 14px 36px rgba(0,0,0,.22); }
    .hero-kicker { color:#86E7F6; text-transform:uppercase; letter-spacing:2px; font-size:.72rem; font-weight:800; }
    .hero-title { color:#F8FCFF; font-size:clamp(1.55rem,2.6vw,2.25rem); font-weight:800; margin:7px 0 5px; }
    .hero-sub { color:#D0E0EE; font-size:.92rem; line-height:1.6; }
    .section-heading { color:#F1F7FF; font-size:1.08rem; font-weight:800; margin:14px 0 12px; }
    .panel { background:linear-gradient(150deg,#142D46,#0E2034); border:1px solid var(--line); border-radius:17px; padding:17px 18px; box-shadow:0 10px 26px rgba(0,0,0,.18); margin-bottom:14px; }
    .panel-title { color:#F5FAFF; font-size:1.0rem; font-weight:800; margin-bottom:10px; }
    .muted { color:#BDD0E1 !important; font-size:.85rem; line-height:1.55; }
    .status-pill { display:inline-block; border-radius:999px; padding:6px 12px; font-size:.78rem; font-weight:800; background:rgba(56,217,150,.13); color:#75F0BA; border:1px solid rgba(56,217,150,.38); }
    .status-pill-warn { background:rgba(255,209,102,.12); color:#FFDF95; border-color:rgba(255,209,102,.36); }
    .small-note { border-left:3px solid #68D8EE; background:rgba(104,216,238,.08); padding:9px 11px; border-radius:0 9px 9px 0; color:#C7DAEA; font-size:.8rem; line-height:1.5; margin-top:10px; }
    [data-testid="stButton"] button { border-radius:10px; font-weight:700; }
    hr { border-color:var(--line) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🛰️ NTN RESEARCH LAB")
    selected_page = st.radio(
        "ĐIỀU HƯỚNG",
        [
            "01 — NTN Simulator",
            "02 — Time-series Link Quality",
            "03 — AI Satellite/Beam & Handover",
        ],
        index=0,
        key="main_navigation",
    )
    st.divider()
    st.markdown("## ⚙️ CẤU HÌNH MÔ PHỎNG")
    st.markdown('<div class="muted">Chỉnh thông số rồi nhấn ÁP DỤNG để lưu.</div>', unsafe_allow_html=True)
    cfg = load_settings()
    with st.form("constellation_settings_form"):
        station_name = st.selectbox("📍 UE / Trạm mặt đất", list(GROUND_STATIONS.keys()), index=list(GROUND_STATIONS.keys()).index(cfg["station_name"]) if cfg["station_name"] in GROUND_STATIONS else 0)
        sim_time = st.slider("⏱️ Thời gian mô phỏng (giây)", 0, 720, int(cfg["sim_time"]), 5)
        altitude_km = st.select_slider("🛰️ Độ cao LEO (km)", options=[500,600,700,850,1000,1200], value=int(cfg["altitude_km"]) if int(cfg["altitude_km"]) in [500,600,700,850,1000,1200] else 600)
        inclination_deg = st.slider("Góc nghiêng quỹ đạo (°)", 30,70,int(cfg["inclination_deg"]),1)
        n_planes = st.slider("Số mặt phẳng quỹ đạo",1,6,int(cfg["n_planes"]),1)
        sats_per_plane = st.slider("Số vệ tinh / mặt phẳng",2,12,int(cfg["sats_per_plane"]),1)
        raan_center = st.slider("RAAN trung tâm (°)",0,359,int(cfg["raan_center"]),1)
        plane_spacing = st.slider("Khoảng cách RAAN giữa các mặt phẳng (°)",1,180,int(cfg["plane_spacing"]),1)
        phase_start = st.slider("Pha ban đầu (°)",0,359,int(cfg["phase_start"]),1)
        apply_config = st.form_submit_button("💾 ÁP DỤNG & LƯU CẤU HÌNH", type="primary", use_container_width=True)
    if apply_config:
        save_settings({**cfg,"station_name":station_name,"sim_time":sim_time,"altitude_km":altitude_km,"inclination_deg":inclination_deg,"n_planes":n_planes,"sats_per_plane":sats_per_plane,"raan_center":raan_center,"plane_spacing":plane_spacing,"phase_start":phase_start})
        # A constellation change invalidates the displayed result from the previous run.
        st.session_state.pop("handover_results", None)
        st.session_state.pop("handover_results_signature", None)
        st.success("Đã lưu settings.json. Kết quả Handover cũ đã được xóa; hãy chạy lại Trang 03.")

    # Restore saved configuration or factory defaults
    restore_col1, restore_col2 = st.columns(2)
    with restore_col1:
        if st.button("↩️ Nạp lại cấu hình đã lưu", use_container_width=True, help="Nạp lại các giá trị đang lưu trong settings.json; bỏ qua thay đổi chưa lưu."):
            st.rerun()
    with restore_col2:
        if st.button("🔄 Khôi phục mặc định", use_container_width=True, help="Ghi lại bộ thông số mặc định của ứng dụng."):
            save_settings(DEFAULT_SETTINGS.copy())
            st.session_state.pop("handover_results", None)
            st.success("Đã khôi phục cấu hình mặc định.")
            st.rerun()
    st.divider()
    st.markdown('<div class="muted">Chòm sao dùng 7 Spot Beams; điều kiện phục vụ Elevation ≥ 10°.</div>', unsafe_allow_html=True)

if selected_page == "01 — NTN Simulator":
    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">NTN 5G-Advanced / 6G · Research Simulator</div>
          <div class="hero-title">🛰️ Mô phỏng chòm sao vệ tinh LEO</div>
          <div class="hero-sub">Trái Đất bản đồ độ phân giải cao · Quỹ đạo Kepler · Visibility · Multi-Beam Link Budget<br>
          <span style="color:#83E6F4;font-weight:800">GIAI ĐOẠN 01</span> — Mô phỏng nhiều vệ tinh, xác định Visibility và Beam tối ưu.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    station_lat, station_lon = GROUND_STATIONS[station_name]
    pos_gs = latlon_to_ecef(station_lat, station_lon)
    satellites = make_constellation(
        n_planes=n_planes,
        sats_per_plane=sats_per_plane,
        altitude_km=float(altitude_km),
        inclination_deg=float(inclination_deg),
        raan_center_deg=float(raan_center),
        plane_spacing_deg=float(plane_spacing),
        phase_start_deg=float(phase_start),
    )
    states = get_satellite_states(satellites, float(sim_time), pos_gs)
    states_sorted = sorted(states, key=lambda item: float(item["elevation_deg"]), reverse=True)
    ready_states = [state for state in states_sorted if bool(state["candidate"])]
    above_horizon_states = [state for state in states_sorted if bool(state["above_horizon"])]
    total_satellites = len(satellites)

    st.markdown('<div class="section-heading">Tổng quan hệ thống tại thời điểm khảo sát</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4, gap="medium")
    k1.metric("Tổng vệ tinh trong chòm sao", f"{total_satellites} / {n_planes * sats_per_plane}")
    k2.metric("Trên đường chân trời UE", f"{len(above_horizon_states)}")
    k3.metric("Đủ điều kiện liên kết", f"{len(ready_states)}")
    k4.metric("Thời gian mô phỏng", f"{sim_time} s", f"{sim_time / 60:.1f} phút")

    st.markdown('<div class="section-heading">🌍 Trái Đất bản đồ 3D · Quỹ đạo LEO · Visibility</div>', unsafe_allow_html=True)
    left, right = st.columns([1.48, 1.0], gap="large")
    with left:
        with st.spinner("Đang dựng Trái Đất bản đồ độ phân giải cao và quỹ đạo vệ tinh..."):
            fig = make_figure(
                satellites=satellites,
                states=states_sorted,
                selected_station=station_name,
                selected_station_pos=pos_gs,
                elapsed_s=float(sim_time),
                station_coords=GROUND_STATIONS,
            )
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displaylogo": False, "scrollZoom": True, "responsive": True, "toImageButtonOptions": {"format": "png", "filename": "NTN_LEO_3D", "scale": 2}},
        )

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">🎯 Trạng thái phục vụ UE & Chọn Beam</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="muted">Trạm đang theo dõi</div><div style="font-size:1.24rem;font-weight:800;color:#FFFFFF;margin:3px 0 13px">{station_name}</div>', unsafe_allow_html=True)
        if ready_states:
            best = max(ready_states, key=lambda item: float(item["elevation_deg"]))
            st.markdown(f'<span class="status-pill">● {len(ready_states)} / {total_satellites} vệ tinh đủ điều kiện</span>', unsafe_allow_html=True)
            st.markdown(
                f'<div style="font-size:.78rem;color:#B9CDDF;margin-top:15px">CẶP (SATELLITE, BEAM) TỐI ƯU</div>'
                f'<div style="font-size:1.30rem;font-weight:800;color:#75DFF0;margin:4px 0 12px;overflow-wrap:anywhere">{best["sat_id"]} · {best["best_beam_id"]}</div>',
                unsafe_allow_html=True,
            )
            m1, m2 = st.columns(2, gap="medium")
            with m1:
                st.metric("Elevation", f'{best["elevation_deg"]:.2f}°')
                st.metric("Khoảng cách đường truyền", f'{best["distance_km"]:,.1f} km')
            with m2:
                st.metric("SNR sau Beam", f'{best["beam_snr_db"]:.2f} dB', f'{best["best_beam_offset_db"]:+.2f} dB roll-off')
                st.metric("Doppler", f'{best["doppler_khz"]:+,.2f} kHz')
            st.markdown(
                f'<div class="small-note">Độ trễ một chiều: <b>{best["delay_ms"]:.2f} ms</b> · '
                f'Góc lệch tâm Beam: <b>{best["best_off_axis_deg"]:.2f}°</b> · '
                f'Công suất thu: <b>{best["rx_power_dbm"]:.2f} dBm</b></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<span class="status-pill status-pill-warn">● Chưa có vệ tinh đủ điều kiện</span>', unsafe_allow_html=True)
            st.warning("Tại thời điểm và vị trí đang chọn, chưa có vệ tinh đạt đồng thời ngưỡng Elevation và công suất thu.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">🧭 Chú giải bản đồ</div>', unsafe_allow_html=True)
        st.markdown('<p>🟢 <b>Vệ tinh đủ điều kiện</b><br><span class="muted">Elevation ≥ 10° và công suất thu đạt ngưỡng.</span></p>', unsafe_allow_html=True)
        st.markdown('<p>🟠 <b>Trên chân trời nhưng chưa đạt</b><br><span class="muted">Nhìn thấy hình học, nhưng chưa đủ điều kiện liên kết.</span></p>', unsafe_allow_html=True)
        st.markdown('<p>⚪ <b>Dưới đường chân trời UE</b><br><span class="muted">Đường truyền thẳng tới trạm bị Trái Đất che khuất.</span></p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-heading">📡 Bảng phân tích Visibility & Beam tối ưu — toàn bộ vệ tinh</div>', unsafe_allow_html=True)
    rows: list[dict[str, Any]] = []
    for state in states_sorted:
        if bool(state["candidate"]):
            status = "ĐỦ ĐIỀU KIỆN"
        elif bool(state["above_horizon"]):
            status = "TRÊN CHÂN TRỜI — CHƯA ĐẠT"
        else:
            status = "DƯỚI ĐƯỜNG CHÂN TRỜI"
        rows.append(
            {
                "Satellite ID": state["sat_id"],
                "Mặt phẳng": int(state["plane"]),
                "RAAN (°)": round(float(state["raan_deg"]), 2),
                "Elevation (°)": round(float(state["elevation_deg"]), 2),
                "Khoảng cách (km)": round(float(state["distance_km"]), 2),
                "Doppler (kHz)": round(float(state["doppler_khz"]), 2),
                "SNR kênh (dB)": round(float(state["snr_db"]), 2),
                "Beam tối ưu": state["best_beam_id"],
                "SNR sau Beam (dB)": round(float(state["beam_snr_db"]), 2),
                "Visibility": status,
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True, height=min(620, 38 + 36 * len(df)))

elif selected_page == "02 — Time-series Link Quality":
    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">NTN 5G-Advanced / 6G · Data Engineering</div>
          <div class="hero-title">📈 Time-series Link Quality</div>
          <div class="hero-sub">Sinh dữ liệu chất lượng liên kết theo thời gian cho toàn bộ vệ tinh trong chòm sao.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-heading">1. Cấu hình tạo Dataset</div>', unsafe_allow_html=True)
    cfg1, cfg2, cfg3, cfg4 = st.columns(4)
    with cfg1:
        scenario_count = st.number_input("Số kịch bản", min_value=1, max_value=30, value=10, step=1)
    with cfg2:
        dataset_duration = st.selectbox("Thời gian mỗi kịch bản (giây)", [120, 300, 600, 720, 1200, 1800], index=3)
    with cfg3:
        sample_interval = st.selectbox("Chu kỳ lấy mẫu (giây)", [1, 2, 5, 10], index=0)
    with cfg4:
        dataset_station = st.selectbox("UE dùng tạo dữ liệu", list(GROUND_STATIONS.keys()), index=list(GROUND_STATIONS.keys()).index(station_name))

    expected_samples = len(np.arange(0, int(dataset_duration) + 1, int(sample_interval)))
    total_satellites_for_dataset = int(n_planes) * int(sats_per_plane)
    expected_rows = int(scenario_count) * expected_samples * total_satellites_for_dataset

    info_cols = st.columns(4)
    info_cols[0].metric("Số vệ tinh / kịch bản", total_satellites_for_dataset)
    info_cols[1].metric("Mốc thời gian / kịch bản", expected_samples)
    info_cols[2].metric("Tổng bản ghi dự kiến", f"{expected_rows:,}")
    info_cols[3].metric("Băng tần link budget", "Ka · 28 GHz")

    if "ntn_dataset" not in st.session_state:
        st.session_state["ntn_dataset"] = None
    if "ntn_dataset_config" not in st.session_state:
        st.session_state["ntn_dataset_config"] = None

    generate_clicked = st.button("🚀 GENERATE DATASET", type="primary", use_container_width=True)
    if generate_clicked:
        station_lat_ds, station_lon_ds = GROUND_STATIONS[dataset_station]
        pos_gs_ds = latlon_to_ecef(station_lat_ds, station_lon_ds)
        time_values = np.arange(0, int(dataset_duration) + 1, int(sample_interval), dtype=int)
        rows: list[dict[str, Any]] = []
        progress = st.progress(0, text="Đang tìm pha quỹ đạo có độ phủ tại UE...")
        status = st.empty()
        try:
            scenario_configs = build_coverage_scenario_configs(
                scenario_count=int(scenario_count),
                raan_center_deg=float(raan_center),
                phase_start_deg=float(phase_start),
                n_planes=int(n_planes),
                sats_per_plane=int(sats_per_plane),
                altitude_km=float(altitude_km),
                inclination_deg=float(inclination_deg),
                plane_spacing_deg=float(plane_spacing),
                time_values=time_values,
                pos_gs=pos_gs_ds,
            )
        except ValueError as exc:
            progress.empty()
            status.empty()
            st.error(str(exc))
            st.stop()

        parts: list[pd.DataFrame] = []
        for scenario_idx, scenario_cfg in enumerate(scenario_configs):
            scenario_phase = float(scenario_cfg["phase_start_deg"])
            scenario_raan = float(scenario_cfg["raan_center_deg"])
            scenario_id = f"SCN_{scenario_idx + 1:02d}"

            # Use the same vectorized physical engine as Page 03.
            # sample_interval_s preserves Page 02's selectable 1/2/5/10-second sampling.
            scenario_ts = simulate_constellation_timeseries(
                n_planes=int(n_planes),
                sats_per_plane=int(sats_per_plane),
                altitude_km=float(altitude_km),
                inclination_deg=float(inclination_deg),
                raan_center_deg=scenario_raan,
                plane_spacing_deg=float(plane_spacing),
                phase_start_deg=scenario_phase,
                station_name=dataset_station,
                duration_s=int(dataset_duration),
                sample_interval_s=int(sample_interval),
            )

            scenario_sats = make_constellation(
                n_planes=int(n_planes),
                sats_per_plane=int(sats_per_plane),
                altitude_km=float(altitude_km),
                inclination_deg=float(inclination_deg),
                raan_center_deg=scenario_raan,
                plane_spacing_deg=float(plane_spacing),
                phase_start_deg=scenario_phase,
            )
            meta_by_sat = {sat["sat_id"]: sat for sat in scenario_sats}
            scenario_ts["scenario_id"] = scenario_id
            scenario_ts["ue_id"] = dataset_station
            scenario_ts["raan_deg"] = scenario_ts["sat_id"].map(
                {sid: float(meta["raan_deg"]) for sid, meta in meta_by_sat.items()}
            )
            scenario_ts["phase_deg"] = scenario_ts["sat_id"].map(
                {sid: float(meta["phase_deg"]) for sid, meta in meta_by_sat.items()}
            )
            scenario_ts["scenario_raan_center_deg"] = scenario_raan
            scenario_ts["scenario_phase_start_deg"] = scenario_phase
            scenario_ts["altitude_km"] = float(altitude_km)
            scenario_ts["inclination_deg"] = float(inclination_deg)
            parts.append(scenario_ts)
            progress.progress(
                (scenario_idx + 1) / int(scenario_count),
                text=f"Hoàn thành {scenario_idx + 1}/{int(scenario_count)} kịch bản (vector hóa + cache)",
            )

        dataset = pd.concat(parts, ignore_index=True)
        # Keep the original dataset fields and append the beam measurements already
        # calculated by the 7-spot physical model; old training scripts can ignore extras.
        dataset = dataset[[
            "scenario_id", "time_s", "ue_id", "sat_id", "plane", "raan_deg", "phase_deg",
            "scenario_raan_center_deg", "scenario_phase_start_deg", "altitude_km", "inclination_deg",
            "elevation_deg", "distance_km", "range_rate_km_s", "doppler_khz", "delay_ms",
            "fspl_db", "rx_power_dbm", "snr_db", "above_horizon", "candidate",
            "best_beam_id", "best_beam_offset_db", "best_off_axis_deg", "beam_snr_db",
        ]].sort_values(["scenario_id", "sat_id", "time_s"], kind="stable").reset_index(drop=True)
        st.session_state["ntn_dataset"] = dataset
        st.session_state["ntn_dataset_config"] = {"scenarios": int(scenario_count), "rows": len(dataset)}
        status.success("Đã tạo Dataset thành công.")
        progress.empty()

    dataset = st.session_state.get("ntn_dataset")
    if dataset is not None and not dataset.empty:
        st.dataframe(dataset.head(40), use_container_width=True, hide_index=True)
        csv_bytes = dataset.to_csv(index=False, float_format="%.8f").encode("utf-8-sig")
        st.download_button(
            "⬇️ DOWNLOAD CSV — NTN LINK QUALITY DATASET",
            data=csv_bytes,
            file_name="ntn_link_quality_dataset.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )

else:
    # -------------------------------------------------------------------------
    # PAGE 03 — STEP 4, 5, 6: AI SATELLITE & BEAM SELECTION + HANDOVER KPI
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">NTN 5G-Advanced / 6G · Intelligent Satellite & Beam Handover Lab</div>
          <div class="hero-title">🤖 AI Satellite/Beam Selection & Handover KPI Comparison</div>
          <div class="hero-sub">Luồng xử lý: UE → Visibility → Danh sách vệ tinh → AI LSTM dự báo SNR (+5s) → Chọn cặp (Satellite_best, Beam_best) → Duy trì / Handover.<br>
          <span style="color:#83E6F4;font-weight:800">GIAI ĐOẠN 04 – 06</span> — Đối chứng KPI và lịch sử chọn vệ tinh/beam theo từng mốc thời gian.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        model_signature = get_lstm_artifact_signature()
        lstm_model, lstm_scalers, lstm_ckpt = load_trained_lstm_bundle(model_signature)
        metrics_file = Path(__file__).resolve().parent / "results" / "step3_lstm" / "metrics.json"
        saved_metrics = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.exists() else {}
        mae_val = saved_metrics.get("LSTM", {}).get("MAE_dB", None)
        rmse_val = saved_metrics.get("LSTM", {}).get("RMSE_dB", None)
        metric_str = (
            f" · Test MAE = {mae_val:.4f} dB, RMSE = {rmse_val:.4f} dB"
            if isinstance(mae_val, (int, float)) and isinstance(rmse_val, (int, float))
            else ""
        )
        st.success(
            f"Đã nạp mô hình AI thành công: `models/lstm_snr.pth` (Lookback = {lstm_ckpt.get('lookback', 30)}s, "
            f"Horizon = +{lstm_ckpt.get('horizon_seconds', 5.0):.0f}s{metric_str})."
        )
    except Exception as exc:
        st.error(f"Không thể nạp mô hình AI: {exc}")
        st.stop()

    st.markdown('<div class="section-heading">1. Tham số thuật toán Handover & Trọng số AI</div>', unsafe_allow_html=True)
    cfg = load_settings()
    ho_duration_options = [300, 600, 720, 1200]
    saved_duration = int(cfg.get("ho_duration", 720))
    safe_duration_index = ho_duration_options.index(saved_duration) if saved_duration in ho_duration_options else ho_duration_options.index(720)
    with st.form("handover_run_form"):
        c1,c2,c3,c4=st.columns(4)
        with c1: ho_duration=st.selectbox("Thời lượng khảo sát (giây)", ho_duration_options, index=safe_duration_index)
        with c2: hyst_margin=st.slider("Hysteresis Margin (dB)",0.5,5.0,float(cfg.get("hyst_margin",2.0)),0.5)
        with c3: ttt_sec=st.slider("Time-to-Trigger — TTT (giây)",1,10,int(cfg.get("ttt_sec",3)),1)
        with c4: ai_weight=st.slider("AI Weight (α)",0.0,1.0,float(cfg.get("ai_weight",0.65)),0.05)
        run_clicked=st.form_submit_button("▶️ CHẠY MÔ PHỎNG",type="primary",use_container_width=True)
    if run_clicked:
        save_settings({**cfg,"ho_duration":int(ho_duration),"hyst_margin":float(hyst_margin),"ttt_sec":int(ttt_sec),"ai_weight":float(ai_weight)})
        st.session_state["handover_run_requested"]=True
    st.info("Thay đổi thông số rồi nhấn CHẠY MÔ PHỎNG. Cấu hình được lưu tự động.")
    current_data_signature = (
        int(n_planes), int(sats_per_plane), float(altitude_km),
        float(inclination_deg), float(raan_center), float(plane_spacing),
        float(phase_start), str(station_name), int(ho_duration),
    )
    if run_clicked:
        data_signature = current_data_signature
        with st.spinner("Đang chạy mô phỏng quỹ đạo, chọn Beam, suy luận LSTM và đánh giá 3 chiến lược Handover..."):
            raw_ts = simulate_constellation_timeseries(
                n_planes=int(n_planes),
                sats_per_plane=int(sats_per_plane),
                altitude_km=float(altitude_km),
                inclination_deg=float(inclination_deg),
                raan_center_deg=float(raan_center),
                plane_spacing_deg=float(plane_spacing),
                phase_start_deg=float(phase_start),
                station_name=station_name,
                duration_s=int(ho_duration),
            )
            # A changed model/scaler file changes model_signature and invalidates AI results.
            pred_ts = cached_lstm_predictions(raw_ts, data_signature, model_signature)

            # Rule-based results are independent of live AI alpha; cache them separately.
            df_max_el, sum_max_el = cached_handover_policy(
                pred_ts, data_signature, model_signature,
                policy="Rule-based (Max Elevation)", hyst_margin_db=0.0, ttt_seconds=1, ai_weight=0.0
            )
            df_hyst, sum_hyst = cached_handover_policy(
                pred_ts, data_signature, model_signature,
                policy="Rule-based + Hysteresis (TTT)", hyst_margin_db=hyst_margin, ttt_seconds=ttt_sec, ai_weight=0.0
            )
            df_ai, sum_ai = cached_handover_policy(
                pred_ts, data_signature, model_signature,
                policy="AI-assisted (LSTM Proactive)", hyst_margin_db=hyst_margin, ttt_seconds=ttt_sec, ai_weight=ai_weight
            )

        st.caption("Tối ưu hiệu năng: quỹ đạo/link budget, suy luận LSTM theo batch và kết quả từng chính sách được cache. Lần đầu với cấu hình mới sẽ tính đầy đủ; đổi riêng α sẽ không mô phỏng lại quỹ đạo.")
        st.session_state["handover_results"]=(raw_ts,pred_ts,df_max_el,sum_max_el,df_hyst,sum_hyst,df_ai,sum_ai,mae_val,rmse_val,data_signature)
    elif "handover_results" in st.session_state:
        saved_result = st.session_state["handover_results"]
        if len(saved_result) >= 11:
            raw_ts,pred_ts,df_max_el,sum_max_el,df_hyst,sum_hyst,df_ai,sum_ai,mae_val,rmse_val,data_signature = saved_result[:11]
        else:  # Backward compatibility with results kept in an older Streamlit session.
            raw_ts,pred_ts,df_max_el,sum_max_el,df_hyst,sum_hyst,df_ai,sum_ai,mae_val,rmse_val = saved_result
            data_signature = current_data_signature
        if tuple(data_signature) != tuple(current_data_signature):
            st.session_state.pop("handover_results", None)
            st.warning("Cấu hình chòm sao hiện tại khác với lần mô phỏng đã lưu. Hãy nhấn CHẠY MÔ PHỎNG để cập nhật KPI.")
            st.stop()
    else:
        st.warning("Hãy nhấn CHẠY MÔ PHỎNG để tạo kết quả. Cấu hình chòm sao cần được áp dụng/lưu trước ở thanh bên.")
        st.stop()

    st.markdown('<div class="section-heading">2. Bảng tổng hợp so sánh KPI (Giai đoạn 06)</div>', unsafe_allow_html=True)
    kpi_table = pd.DataFrame([sum_max_el, sum_hyst, sum_ai])
    st.dataframe(kpi_table, use_container_width=True, hide_index=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Satellite Handover (AI)",
        f"{sum_ai['Satellite Handover']} lần",
        f"{sum_ai['Satellite Handover'] - sum_max_el['Satellite Handover']:+d} so với Max-El",
    )
    c2.metric(
        "Throughput TB (AI)",
        f"{sum_ai['Throughput TB (Mbps)']:.2f} Mbps",
        f"{sum_ai['Throughput TB (Mbps)'] - sum_hyst['Throughput TB (Mbps)']:+.2f} Mbps so với Hysteresis",
    )
    c3.metric(
        "Link Availability (AI)",
        f"{sum_ai['Link Availability (%)']:.2f}%",
        f"Outage Ratio: {sum_ai['Outage Ratio (%)']:.2f}%",
    )
    if isinstance(mae_val, (int, float)) and isinstance(rmse_val, (int, float)):
        c4.metric("Độ chính xác AI (Test Set)", f"MAE: {mae_val:.4f} dB", f"RMSE: {rmse_val:.4f} dB")
    else:
        c4.metric("Tỷ lệ Ping-Pong (AI)", f"{sum_ai['Ping-Pong (%)']:.2f}%")

    # Bang kiem chung do nhay AI Weight (0 vs 0.65 vs 1.0)
    with st.expander("🔬 Kiểm chứng tự động độ nhạy tham số AI Weight (α = 0 vs 0.65 vs 1.0)", expanded=False):
        sensitivity_table = cached_alpha_sensitivity(pred_ts, data_signature, model_signature, hyst_margin, ttt_sec)
        st.dataframe(sensitivity_table, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-heading">3. Lịch sử lựa chọn cặp (Satellite, Beam) theo thời gian (Bước 3)</div>', unsafe_allow_html=True)
    hist_step = st.selectbox("Bước nhảy hiển thị bảng lịch sử (giây)", [1, 5, 10, 20, 30], index=2)
    history_df = pd.DataFrame(
        {
            "Thời gian (s)": df_ai["time_s"],
            "Rule-based (Max-El)": df_max_el["sat_beam_pair"],
            "Hysteresis (TTT)": df_hyst["sat_beam_pair"],
            "AI LSTM (Sat / Beam)": df_ai["sat_beam_pair"],
            "SNR sau Beam (dB)": df_ai["beam_snr_db"].round(2),
            "SNR dự báo +5s (dB)": df_ai["pred_snr_5s_db"].round(2),
            "Throughput AI (Mbps)": df_ai["throughput_mbps"].round(1),
            "Sự kiện AI": np.where(
                df_ai["handover"],
                "🔄 SAT HANDOVER",
                np.where(df_ai["beam_switch"], "📡 BEAM SWITCH", "Duy trì"),
            ),
        }
    )
    # Hien thi tat ca cac moc theo buoc nhay + tat ca cac giay xay ra Handover cua AI hoac Rule-based
    mask_show = (history_df["Thời gian (s)"] % int(hist_step) == 0) | df_ai["handover"] | df_max_el["handover"]
    st.dataframe(history_df[mask_show], use_container_width=True, hide_index=True, height=380)

    st.markdown('<div class="section-heading">4. Biểu đồ diễn biến SNR sau Beam & Lịch trình chọn vệ tinh</div>', unsafe_allow_html=True)
    fig_snr = go.Figure()
    fig_snr.add_trace(
        go.Scatter(
            x=df_max_el["time_s"], y=df_max_el["beam_snr_db"],
            mode="lines", name="Rule-based (Max Elevation)",
            line=dict(color="#F4B95F", width=1.8, dash="dot"),
        )
    )
    fig_snr.add_trace(
        go.Scatter(
            x=df_hyst["time_s"], y=df_hyst["beam_snr_db"],
            mode="lines", name="Rule-based + Hysteresis (TTT)",
            line=dict(color="#B69CFF", width=2.0),
        )
    )
    fig_snr.add_trace(
        go.Scatter(
            x=df_ai["time_s"], y=df_ai["beam_snr_db"],
            mode="lines", name="AI-assisted (LSTM Proactive)",
            line=dict(color="#35E39A", width=2.6),
        )
    )
    ai_ho_points = df_ai[df_ai["handover"]]
    if not ai_ho_points.empty:
        fig_snr.add_trace(
            go.Scatter(
                x=ai_ho_points["time_s"], y=ai_ho_points["beam_snr_db"],
                mode="markers", name="Điểm chuyển giao AI Handover",
                marker=dict(color="#FFD166", size=10, symbol="diamond", line=dict(color="#081321", width=1.2)),
                text=[f"t={int(r.time_s)}s → {r.sat_beam_pair}" for r in ai_ho_points.itertuples()],
                hovertemplate="%{text}<br>SNR sau Beam: %{y:.2f} dB<extra></extra>",
            )
        )
    fig_snr.update_layout(
        title=f"So sánh SNR liên kết sau chọn Beam tại {station_name}",
        height=390,
        margin=dict(l=20, r=20, t=55, b=35),
        paper_bgcolor="#10243A",
        plot_bgcolor="#0B1B2B",
        font=dict(color="#F0F6FF", family="Be Vietnam Pro, Arial"),
        xaxis=dict(title="Thời gian mô phỏng (s)", gridcolor="rgba(180,210,235,.13)"),
        yaxis=dict(title="Effective SNR sau Beam (dB)", gridcolor="rgba(180,210,235,.13)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig_snr, use_container_width=True, config={"displaylogo": False, "responsive": True, "staticPlot": False})

    fig_sat = go.Figure()
    all_sats_sorted = sorted(pred_ts["sat_id"].unique().tolist())
    sat_to_idx = {sid: idx for idx, sid in enumerate(all_sats_sorted)}
    for label, sub_df, color in [
        ("Rule-based (Max Elevation)", df_max_el, "#F4B95F"),
        ("Rule-based + Hysteresis", df_hyst, "#B69CFF"),
        ("AI-assisted (LSTM)", df_ai, "#35E39A"),
    ]:
        valid = sub_df[sub_df["serving_sat"] != "OUTAGE"].copy()
        fig_sat.add_trace(
            go.Scatter(
                x=valid["time_s"],
                y=[sat_to_idx[s] for s in valid["serving_sat"]],
                mode="markers",
                name=label,
                marker=dict(size=5, color=color, opacity=0.85),
                text=valid["sat_beam_pair"],
                hovertemplate="t=%{x}s · Cặp chọn: %{text}<extra></extra>",
            )
        )
    fig_sat.update_layout(
        title="Lịch trình lựa chọn vệ tinh phục vụ (Serving Satellite Timeline)",
        height=350,
        margin=dict(l=20, r=20, t=55, b=35),
        paper_bgcolor="#10243A",
        plot_bgcolor="#0B1B2B",
        font=dict(color="#F0F6FF", family="Be Vietnam Pro, Arial"),
        xaxis=dict(title="Thời gian mô phỏng (s)", gridcolor="rgba(180,210,235,.13)"),
        yaxis=dict(
            title="Satellite ID",
            tickmode="array",
            tickvals=list(range(len(all_sats_sorted))),
            ticktext=all_sats_sorted,
            gridcolor="rgba(180,210,235,.13)",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig_sat, use_container_width=True, config={"displaylogo": False, "responsive": True})

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "⬇️ TẢI BẢNG SO SÁNH KPI (CSV)",
            data=kpi_table.to_csv(index=False).encode("utf-8-sig"),
            file_name="ntn_handover_kpi_comparison.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_dl2:
        st.download_button(
            "⬇️ TẢI LỊCH SỬ LỰA CHỌN SATELLITE/BEAM (CSV)",
            data=history_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="ntn_satellite_beam_selection_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown(
        '<div style="text-align:center;color:#8FA8BE;font-size:.78rem;padding:22px 0 5px">NTN LEO Research Simulator · Step 1–6 — Earth 3D → Visibility → AI LSTM → Satellite/Beam Handover → KPI</div>',
        unsafe_allow_html=True,
    )
