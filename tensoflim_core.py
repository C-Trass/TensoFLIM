
"""
tensoflim_core.py

Core computation utilities for TensoFLIM:
- Load per-pixel lifetime TIFFs (float or integer)
- Compute apparent FRET efficiency relative to a reference lifetime tau0
- Convert efficiency -> force via a calibration model
- Zero (offset-correct) force using the tensionless control distribution
- Save analysis-grade float32 TIFFs + optional 16-bit preview TIFFs
- Export summary statistics to JSON/TXT

Designed to be used by a GUI front-end (Tkinter/PySide) or CLI.

Author: Conor A. Treacy (project), code assembled with ChatGPT
License: choose later (MIT/BSD recommended)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple, Literal, Any

import numpy as np

try:
    import tifffile as tiff
except Exception:  # pragma: no cover
    tiff = None

# ---------------------------
# I/O
# ---------------------------

def load_lifetime_tiff(path: str | Path) -> np.ndarray:
    """
    Load a per-pixel lifetime map from a TIFF. Supports float32/float64/uint16/etc.

    Returns
    -------
    tau : np.ndarray (2D)
        Lifetime map, dtype float32, NaNs preserved.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    if tiff is None:
        raise ImportError("tifffile is required to load float TIFFs. Please `pip install tifffile`.")

    arr = tiff.imread(str(p))
    arr = np.asarray(arr)

    # If it's a stack, try to squeeze to 2D.
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D lifetime map, got shape {arr.shape} from {p.name}")

    return arr.astype(np.float32, copy=False)


def save_tiff_float32(path: str | Path, arr: np.ndarray) -> None:
    """Save float32 TIFF (analysis-grade)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if tiff is None:
        raise ImportError("tifffile is required to save float TIFFs. Please `pip install tifffile`.")
    a = np.asarray(arr, dtype=np.float32)
    tiff.imwrite(str(p), a, dtype=np.float32)


def robust_range_nonzero(arr: np.ndarray, low: float = 1.0, high: float = 99.0) -> Tuple[float, float]:
    """
    Percentile range ignoring NaN/Inf and ignoring zeros.
    Useful for preview images so they do not look empty.
    """
    a = np.asarray(arr, dtype=np.float32)
    a = a[np.isfinite(a)]
    a = a[a != 0]
    if a.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(a, low), np.percentile(a, high)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return float(vmin), float(vmax)


def to_uint16(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Scale to 0..65535 for preview."""
    a = np.asarray(arr, dtype=np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    scaled = (a - vmin) / (vmax - vmin)
    scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled * 65535.0).astype(np.uint16)


def save_tiff_preview_uint16(path: str | Path, arr: np.ndarray, low: float = 1.0, high: float = 99.0) -> Tuple[float, float]:
    """
    Save a 16-bit preview TIFF with robust percentile scaling.

    Returns
    -------
    (vmin, vmax) used for scaling
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if tiff is None:
        raise ImportError("tifffile is required to save TIFFs. Please `pip install tifffile`.")
    vmin, vmax = robust_range_nonzero(arr, low=low, high=high)
    img16 = to_uint16(arr, vmin, vmax)
    tiff.imwrite(str(p), img16, dtype=np.uint16)
    return vmin, vmax


# ---------------------------
# Calibration models
# ---------------------------

@dataclass(frozen=True)
class CalibrationModel:
    """
    Base calibration model.

    You can add additional models later without changing GUI code by:
    - Creating a new dataclass with the same interface
    - Registering it in BUILTIN_MODELS
    """
    name: str

    def force_from_efficiency(self, E: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class LinearSpringForsterModel(CalibrationModel):
    """
    A simple, physically interpretable model:

    1) Convert apparent efficiency E to distance r using Förster relation:
         r = R0 * ((1/E) - 1)^(1/6)

    2) Convert distance to force using a linear spring:
         F = k * (r - r0)

    Notes
    -----
    - This is a pragmatic default. Many published tension sensors use
      a non-linear calibration (often WLC-like). You can replace this
      model with your preferred calibration.
    - Handles E <= 0 by mapping to a capped large distance r_cap_nm.
    - Handles E >= 1 by mapping to r_min_nm (near zero).

    Parameters
    ----------
    R0_nm : Förster radius (nm)
    r0_nm : rest distance at 0 force (nm)
    k_pN_per_nm : spring constant in pN/nm (so output is pN)
    r_cap_nm : cap distance for E<=0 (nm)
    r_min_nm : minimum distance for E>=1 (nm)
    """
    R0_nm: float
    r0_nm: float
    k_pN_per_nm: float
    r_cap_nm: float = 15.0
    r_min_nm: float = 1.0

    def force_from_efficiency(self, E: np.ndarray) -> np.ndarray:
        E = np.asarray(E, dtype=np.float32)

        # Compute r in nm with safe handling
        r = np.empty_like(E, dtype=np.float32)

        # E <= 0 -> very low FRET, far distance
        mask_low = E <= 0
        r[mask_low] = np.float32(self.r_cap_nm)

        # E >= 1 -> extremely high FRET, tiny distance
        mask_high = E >= 1
        r[mask_high] = np.float32(self.r_min_nm)

        # 0 < E < 1 -> standard Förster inversion
        mask_mid = (~mask_low) & (~mask_high)
        Em = E[mask_mid]
        r[mask_mid] = np.float32(self.R0_nm) * np.power((1.0 / Em) - 1.0, 1.0 / 6.0)

        # Linear spring (pN)
        F = np.float32(self.k_pN_per_nm) * (r - np.float32(self.r0_nm))
        return F


# A minimal registry of built-ins, you can add your own calibrations here.
BUILTIN_MODELS: Dict[str, CalibrationModel] = {
    # Placeholder defaults. Replace with your real parameters when ready.
    # Example values only.
    "LinearSpring_Default": LinearSpringForsterModel(
        name="LinearSpring_Default",
        R0_nm=5.99,     # example
        r0_nm=7.5,      # example
        k_pN_per_nm=0.5 # example
    ),
}


# ---------------------------
# Core maths
# ---------------------------

ZeroMethod = Literal["mean", "median"]

def compute_tau0(control_tau: np.ndarray, method: ZeroMethod = "mean") -> float:
    """
    Compute tau0 reference from the control lifetime image.
    Assumes the image is already thresholded/masked by the user.
    """
    a = np.asarray(control_tau, dtype=np.float32)
    a = a[np.isfinite(a)]
    if a.size == 0:
        raise ValueError("Control lifetime image contains no finite pixels.")
    if method == "mean":
        return float(a.mean())
    if method == "median":
        return float(np.median(a))
    raise ValueError(f"Unknown method: {method}")


def efficiency_from_tau(tau: np.ndarray, tau0: float) -> np.ndarray:
    """
    Apparent efficiency relative to tau0:
        E = 1 - tau/tau0

    This matches your stated convention (VincTL defines 0).
    """
    tau = np.asarray(tau, dtype=np.float32)
    if tau0 <= 0:
        raise ValueError("tau0 must be > 0.")
    return 1.0 - (tau / np.float32(tau0))


def summarize(arr: np.ndarray) -> Dict[str, float]:
    """Basic summary stats ignoring NaN/Inf."""
    a = np.asarray(arr, dtype=np.float32)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "p05": float(np.percentile(a, 5)),
        "p95": float(np.percentile(a, 95)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


@dataclass
class TensoFLIMResult:
    tau0_ns: float
    zero_method: str
    offset_force_pN: float
    model: Dict[str, Any]
    control_stats: Dict[str, float]
    test_stats: Dict[str, float]


def compute_force_maps(
    control_tau: np.ndarray,
    test_tau: np.ndarray,
    model: CalibrationModel,
    tau0_method: ZeroMethod = "mean",
    force_zero_method: ZeroMethod = "median",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, TensoFLIMResult]:
    """
    Compute efficiency and force maps for control and test, then offset-correct force
    using the control distribution (Strategy B).

    Returns
    -------
    E_control, E_test, F_control_zeroed, F_test_zeroed, result_summary
    """
    if control_tau.shape != test_tau.shape:
        raise ValueError(f"Control and test images must have same shape, got {control_tau.shape} vs {test_tau.shape}")

    tau0 = compute_tau0(control_tau, method=tau0_method)

    E_control = efficiency_from_tau(control_tau, tau0)
    E_test = efficiency_from_tau(test_tau, tau0)

    F_control = model.force_from_efficiency(E_control)
    F_test = model.force_from_efficiency(E_test)

    # Offset-correct in force-space so VincTL is centred at 0 pN
    if force_zero_method == "mean":
        offset = float(np.nanmean(F_control))
    elif force_zero_method == "median":
        offset = float(np.nanmedian(F_control))
    else:
        raise ValueError(f"Unknown force_zero_method: {force_zero_method}")

    F_control_z = F_control - np.float32(offset)
    F_test_z = F_test - np.float32(offset)

    summary = TensoFLIMResult(
        tau0_ns=float(tau0 * 1e9),
        zero_method=f"tau0={tau0_method}, force_offset={force_zero_method}",
        offset_force_pN=offset,
        model=asdict(model) if hasattr(model, "__dataclass_fields__") else {"name": getattr(model, "name", "model")},
        control_stats={
            "efficiency": summarize(E_control),
            "force_pN_zeroed": summarize(F_control_z),
            "force_pN_raw": summarize(F_control),
        },
        test_stats={
            "efficiency": summarize(E_test),
            "force_pN_zeroed": summarize(F_test_z),
            "force_pN_raw": summarize(F_test),
        },
    )

    return E_control, E_test, F_control_z, F_test_z, summary


def export_results(
    out_dir: str | Path,
    control_tau: np.ndarray,
    test_tau: np.ndarray,
    E_control: np.ndarray,
    E_test: np.ndarray,
    F_control_z: np.ndarray,
    F_test_z: np.ndarray,
    summary: TensoFLIMResult,
    save_previews: bool = True,
) -> Dict[str, str]:
    """
    Save float32 maps and a summary JSON. Optionally save 16-bit previews.

    Returns dict of saved file paths (strings).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    saved: Dict[str, str] = {}

    # Float32 analysis-grade outputs
    maps = {
        "control_tau_ns_float32.tif": control_tau,
        "test_tau_ns_float32.tif": test_tau,
        "control_E_float32.tif": E_control,
        "test_E_float32.tif": E_test,
        "control_force_pN_zeroed_float32.tif": F_control_z,
        "test_force_pN_zeroed_float32.tif": F_test_z,
    }
    for fname, arr in maps.items():
        fp = out / fname
        save_tiff_float32(fp, arr)
        saved[fname] = str(fp)

    # Summary JSON
    import json
    summary_path = out / "tensoflim_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)
    saved["tensoflim_summary.json"] = str(summary_path)

    # Human-readable TXT
    txt_path = out / "tensoflim_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"tau0 (ns): {summary.tau0_ns:.6f}\n")
        f.write(f"force offset (pN): {summary.offset_force_pN:.6f}\n")
        f.write(f"model: {summary.model.get('name','')}\n\n")
        f.write("Control (zeroed force) stats:\n")
        for k, v in summary.control_stats["force_pN_zeroed"].items():
            f.write(f"  {k}: {v}\n")
        f.write("\nTest (zeroed force) stats:\n")
        for k, v in summary.test_stats["force_pN_zeroed"].items():
            f.write(f"  {k}: {v}\n")
    saved["tensoflim_summary.txt"] = str(txt_path)

    # Optional 16-bit previews (for quick viewing in ImageJ/Fiji)
    if save_previews:
        preview_items = {
            "control_force_pN_zeroed_preview_uint16.tif": F_control_z,
            "test_force_pN_zeroed_preview_uint16.tif": F_test_z,
        }
        for fname, arr in preview_items.items():
            fp = out / fname
            vmin, vmax = save_tiff_preview_uint16(fp, arr, low=1, high=99)
            saved[fname] = str(fp)
            # Log scaling used
            saved[fname + "_scaling"] = f"vmin={vmin:.6g}, vmax={vmax:.6g}"

    return saved


def run_tensoflim(
    control_tiff: str | Path,
    test_tiff: str | Path,
    out_dir: str | Path,
    model_name: str = "LinearSpring_Default",
    tau0_method: ZeroMethod = "mean",
    force_zero_method: ZeroMethod = "median",
    save_previews: bool = True,
) -> Dict[str, str]:
    """
    Convenience one-shot runner used by a GUI.

    Assumes the TIFFs are already lifetime maps in ns (as you stated).
    If your maps are in seconds, convert before saving or adjust here.
    """
    control_tau = load_lifetime_tiff(control_tiff)
    test_tau = load_lifetime_tiff(test_tiff)

    # If the TIFF lifetime maps are in ns (typical), keep them.
    # If you store lifetimes in seconds, multiply by 1e9 here.

    if model_name not in BUILTIN_MODELS:
        raise ValueError(f"Unknown model_name '{model_name}'. Available: {list(BUILTIN_MODELS.keys())}")

    model = BUILTIN_MODELS[model_name]

    E_c, E_t, F_cz, F_tz, summary = compute_force_maps(
        control_tau=control_tau,
        test_tau=test_tau,
        model=model,
        tau0_method=tau0_method,
        force_zero_method=force_zero_method,
    )

    return export_results(
        out_dir=out_dir,
        control_tau=control_tau,
        test_tau=test_tau,
        E_control=E_c,
        E_test=E_t,
        F_control_z=F_cz,
        F_test_z=F_tz,
        summary=summary,
        save_previews=save_previews,
    )


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="TensoFLIM core runner (no GUI).")
    parser.add_argument("--control", required=True, help="Control (VincTL) lifetime TIFF")
    parser.add_argument("--test", required=True, help="Test (VincTS) lifetime TIFF")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--model", default="LinearSpring_Default", help=f"Calibration model: {list(BUILTIN_MODELS.keys())}")
    parser.add_argument("--tau0_method", default="mean", choices=["mean", "median"])
    parser.add_argument("--force_zero_method", default="median", choices=["mean", "median"])
    parser.add_argument("--no_previews", action="store_true", help="Disable 16-bit preview TIFFs")
    args = parser.parse_args()

    saved = run_tensoflim(
        control_tiff=args.control,
        test_tiff=args.test,
        out_dir=args.out,
        model_name=args.model,
        tau0_method=args.tau0_method,
        force_zero_method=args.force_zero_method,
        save_previews=not args.no_previews,
    )
    print("Saved:")
    for k, v in saved.items():
        print(f"  {k}: {v}")
