# ============================================================
# Core-shell nanostructure Keras + material-library Streamlit app
# Required files:
#   best_model.keras
#   x_scaler.pkl
#   y_scaler.pkl
#   training_domain.json
#   requirements.txt
# ============================================================

from __future__ import annotations

import os
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import yaml
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tensorflow.keras.models import load_model


# ============================================================
# 0) Streamlit page
# ============================================================

st.set_page_config(
    page_title="Core-shell Cross-Section Predictor",
    page_icon="🔬",
    layout="wide",
)

st.title("Core–Shell Nanostructure Cross-Section Predictor")

st.markdown(
    """
This app predicts **scattering**, **absorption**, and **extinction** spectra
for coupled core–shell nanostructures using your trained neural-network model
and wavelength-dependent material data from refractiveindex.info.
"""
)


# ============================================================
# 1) Paths and constants
# ============================================================

APP_DIR = Path(__file__).resolve().parent

MODEL_PATH = APP_DIR / "best_model.keras"
X_SCALER_PATH = APP_DIR / "x_scaler.pkl"
Y_SCALER_PATH = APP_DIR / "y_scaler.pkl"
DOMAIN_PATH = APP_DIR / "training_domain.json"

RI_DB_DIR = APP_DIR / "refractiveindex.info-database"
MAIN_PATH = RI_DB_DIR / "database" / "data" / "main"

INPUT_COLS = [
    "r0", "d", "g", "n0", "lda0",
    "eps_c1", "eps_c2", "eps_s1", "eps_s2",
]


# ============================================================
# 2) Download refractiveindex.info database
# ============================================================

@st.cache_resource(show_spinner=False)
def download_refractive_index_database():
    if MAIN_PATH.exists():
        return

    if shutil.which("git") is None:
        raise RuntimeError(
            "Git is not available. Add refractiveindex.info-database manually "
            "or add git support in deployment."
        )

    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/polyanskiy/refractiveindex.info-database.git",
            str(RI_DB_DIR),
        ],
        check=True,
    )


# ============================================================
# 3) Material database functions
# ============================================================

@st.cache_data(show_spinner=False)
def scan_material_database() -> Dict[str, list[str]]:
    download_refractive_index_database()

    material_db: Dict[str, list[str]] = {}

    for material_path in sorted(MAIN_PATH.iterdir()):
        if not material_path.is_dir():
            continue

        yaml_files = []

        for root, _, files in os.walk(material_path):
            for file in files:
                if file.endswith((".yml", ".yaml")):
                    yaml_files.append(str(Path(root) / file))

        if yaml_files:
            material_db[material_path.name] = sorted(yaml_files)

    return material_db


def source_dict(material_db: Dict[str, list[str]], material: str) -> Dict[str, str]:
    files = material_db[material]
    names = [Path(f).stem for f in files]
    return dict(zip(names, files))


def preferred_source(material_db: Dict[str, list[str]], material: str) -> str:
    sources = list(source_dict(material_db, material).keys())

    for keyword in ["Johnson", "Rakic", "Palik", "Babar", "Werner", "Olmon", "Christy"]:
        for source in sources:
            if keyword.lower() in source.lower():
                return source

    return sources[0]


@st.cache_data(show_spinner=False)
def parse_yaml_to_epsilon(filename: str) -> pd.DataFrame:
    with open(filename, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    rows = []

    if not data or "DATA" not in data:
        return pd.DataFrame()

    for block in data["DATA"]:
        block_type = block.get("type", "")
        raw_data = block.get("data", "")

        if not raw_data:
            continue

        if block_type == "tabulated nk":
            for line in raw_data.strip().split("\n"):
                vals = line.split()
                if len(vals) < 3:
                    continue

                wl_nm = float(vals[0]) * 1000.0
                n = float(vals[1])
                k = float(vals[2])

                eps_real = n**2 - k**2
                eps_imag = 2.0 * n * k

                rows.append([wl_nm, n, k, eps_real, eps_imag])

        elif block_type == "tabulated n":
            for line in raw_data.strip().split("\n"):
                vals = line.split()
                if len(vals) < 2:
                    continue

                wl_nm = float(vals[0]) * 1000.0
                n = float(vals[1])
                k = 0.0

                eps_real = n**2
                eps_imag = 0.0

                rows.append([wl_nm, n, k, eps_real, eps_imag])

    df = pd.DataFrame(
        rows,
        columns=["wavelength_nm", "n", "k", "eps_real", "eps_imag"]
    )

    if df.empty:
        return df

    return (
        df.sort_values("wavelength_nm")
        .drop_duplicates("wavelength_nm")
        .reset_index(drop=True)
    )


def epsilon_interpolator(material_db: Dict[str, list[str]], material: str, source: str):
    sources = source_dict(material_db, material)
    df_eps = parse_yaml_to_epsilon(sources[source])

    if df_eps.empty or len(df_eps) < 2:
        raise ValueError(f"No valid optical data found for {material}: {source}")

    wl = df_eps["wavelength_nm"].to_numpy()
    eps_real = df_eps["eps_real"].to_numpy()
    eps_imag = df_eps["eps_imag"].to_numpy()

    eps_real_fun = interp1d(
        wl, eps_real,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate"
    )

    eps_imag_fun = interp1d(
        wl, eps_imag,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate"
    )

    return eps_real_fun, eps_imag_fun, df_eps


# ============================================================
# 4) Load model, scalers, domain
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model_and_scalers():
    missing = []

    for p in [MODEL_PATH, X_SCALER_PATH, Y_SCALER_PATH, DOMAIN_PATH]:
        if not p.exists():
            missing.append(p.name)

    if missing:
        raise FileNotFoundError(
            "Missing files: " + ", ".join(missing)
        )

    model = load_model(MODEL_PATH, compile=False)
    x_scaler = joblib.load(X_SCALER_PATH)
    y_scaler = joblib.load(Y_SCALER_PATH)

    with open(DOMAIN_PATH, "r") as f:
        domain = json.load(f)

    return model, x_scaler, y_scaler, domain


# ============================================================
# 5) Prediction functions
# ============================================================

def check_training_domain(X: pd.DataFrame, domain: dict) -> list[str]:
    warnings = []

    for col in INPUT_COLS:
        qmin = float(X[col].min())
        qmax = float(X[col].max())

        dmin = float(domain[col]["min"])
        dmax = float(domain[col]["max"])

        if qmin < dmin or qmax > dmax:
            warnings.append(
                f"{col}: selected range [{qmin:.4g}, {qmax:.4g}] "
                f"is outside training range [{dmin:.4g}, {dmax:.4g}]"
            )

    return warnings


def predict_cross_sections(model, x_scaler, y_scaler, X_input_raw: np.ndarray):
    Xs = x_scaler.transform(
        np.asarray(X_input_raw, dtype=np.float32)
    ).astype(np.float32)

    y_scaled = model.predict(Xs, verbose=0)

    y_log = y_scaler.inverse_transform(y_scaled)

    y = 10 ** y_log
    y = np.maximum(y, 0.0)

    C_scatt = y[:, 0]
    C_absor = y[:, 1]
    C_ext = y[:, 2]

    return C_scatt, C_absor, C_ext


def predict_material_spectrum(
    material_db: Dict[str, list[str]],
    model,
    x_scaler,
    y_scaler,
    domain,
    r0: float,
    d: float,
    g: float,
    n0: float,
    core_material: str,
    core_source: str,
    shell_material: str,
    shell_source: str,
    lambda_min: float,
    lambda_max: float,
    lambda_step: float,
) -> tuple[pd.DataFrame, list[str]]:

    if lambda_step <= 0:
        raise ValueError("Wavelength step must be positive.")

    if lambda_max <= lambda_min:
        raise ValueError("Maximum wavelength must be larger than minimum wavelength.")

    wavelengths = np.arange(lambda_min, lambda_max + 0.5 * lambda_step, lambda_step)

    core_eps1_fun, core_eps2_fun, _ = epsilon_interpolator(
        material_db, core_material, core_source
    )

    shell_eps1_fun, shell_eps2_fun, _ = epsilon_interpolator(
        material_db, shell_material, shell_source
    )

    eps_c1 = core_eps1_fun(wavelengths)
    eps_c2 = core_eps2_fun(wavelengths)
    eps_s1 = shell_eps1_fun(wavelengths)
    eps_s2 = shell_eps2_fun(wavelengths)

    X_sweep = pd.DataFrame({
        "r0": r0,
        "d": d,
        "g": g,
        "n0": n0,
        "lda0": wavelengths,
        "eps_c1": eps_c1,
        "eps_c2": eps_c2,
        "eps_s1": eps_s1,
        "eps_s2": eps_s2,
    })[INPUT_COLS]

    warnings = check_training_domain(X_sweep, domain)

    C_scatt, C_absor, C_ext = predict_cross_sections(
        model,
        x_scaler,
        y_scaler,
        X_sweep.to_numpy()
    )

    out = X_sweep.copy()
    out["C_scatt_total"] = C_scatt
    out["C_absor_total"] = C_absor
    out["C_ext_total_model"] = C_ext
    out["C_ext_total_sum"] = C_scatt + C_absor

    return out, warnings


# ============================================================
# 6) Plotting
# ============================================================

def plot_cross_sections(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(df["lda0"], df["C_scatt_total"], linewidth=2, label="C_scatt_total")
    ax.plot(df["lda0"], df["C_absor_total"], linewidth=2, label="C_absor_total")
    ax.plot(df["lda0"], df["C_ext_total_model"], linewidth=2, label="C_ext_total model")
    ax.plot(df["lda0"], df["C_ext_total_sum"], "--", linewidth=2, label="C_scatt + C_absor")

    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Cross section")
    ax.set_title("Predicted cross-section spectra")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    return fig


def plot_permittivity(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(df["lda0"], df["eps_c1"], linewidth=2, label="Core Re(ε)")
    ax.plot(df["lda0"], df["eps_c2"], linewidth=2, label="Core Im(ε)")
    ax.plot(df["lda0"], df["eps_s1"], "--", linewidth=2, label="Shell Re(ε)")
    ax.plot(df["lda0"], df["eps_s2"], "--", linewidth=2, label="Shell Im(ε)")

    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Permittivity")
    ax.set_title("Material permittivity")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    return fig


# ============================================================
# 7) Load resources
# ============================================================

with st.spinner("Loading model and material database..."):
    try:
        material_db = scan_material_database()
        materials = sorted(material_db.keys())
        model, x_scaler, y_scaler, domain = load_model_and_scalers()
    except Exception as exc:
        st.error(str(exc))
        st.stop()


# ============================================================
# 8) Sidebar
# ============================================================

with st.sidebar:
    st.header("Geometry")

    r0 = st.number_input(
        "Core radius r0 [nm]",
        min_value=5.0,
        max_value=100.0,
        value=75.0,
        step=1.0
    )

    d = st.number_input(
        "Shell thickness d [nm]",
        min_value=1.0,
        max_value=50.0,
        value=10.0,
        step=1.0
    )

    g = st.number_input(
        "Gap g [nm]",
        min_value=2.0,
        max_value=50.0,
        value=10.0,
        step=1.0
    )

    n0 = st.number_input(
        "Surrounding refractive index n0",
        min_value=1.0,
        max_value=1.52,
        value=1.35,
        step=0.01
    )

    st.header("Wavelength sweep")

    lambda_min = st.number_input(
        "Minimum wavelength [nm]",
        min_value=200.0,
        max_value=1000.0,
        value=400.0,
        step=10.0
    )

    lambda_max = st.number_input(
        "Maximum wavelength [nm]",
        min_value=200.0,
        max_value=1000.0,
        value=900.0,
        step=10.0
    )

    lambda_step = st.number_input(
        "Wavelength step [nm]",
        min_value=1.0,
        max_value=50.0,
        value=5.0,
        step=1.0
    )

    st.header("Materials")

    default_core = "Au" if "Au" in materials else materials[0]
    default_shell = "Ag" if "Ag" in materials else materials[0]

    core_material = st.selectbox(
        "Core material",
        materials,
        index=materials.index(default_core)
    )

    core_sources = list(source_dict(material_db, core_material).keys())
    core_default_source = preferred_source(material_db, core_material)

    core_source = st.selectbox(
        "Core data source",
        core_sources,
        index=core_sources.index(core_default_source)
    )

    shell_material = st.selectbox(
        "Shell material",
        materials,
        index=materials.index(default_shell)
    )

    shell_sources = list(source_dict(material_db, shell_material).keys())
    shell_default_source = preferred_source(material_db, shell_material)

    shell_source = st.selectbox(
        "Shell data source",
        shell_sources,
        index=shell_sources.index(shell_default_source)
    )

    run_prediction = st.button(
        "Predict spectrum",
        type="primary",
        use_container_width=True
    )


# ============================================================
# 9) Main app
# ============================================================

if run_prediction:
    with st.spinner("Predicting spectrum..."):
        try:
            df, warnings = predict_material_spectrum(
                material_db=material_db,
                model=model,
                x_scaler=x_scaler,
                y_scaler=y_scaler,
                domain=domain,
                r0=r0,
                d=d,
                g=g,
                n0=n0,
                core_material=core_material,
                core_source=core_source,
                shell_material=shell_material,
                shell_source=shell_source,
                lambda_min=lambda_min,
                lambda_max=lambda_max,
                lambda_step=lambda_step,
            )

        except Exception as exc:
            st.error(str(exc))
            st.stop()

    st.success("Prediction completed.")

    if warnings:
        st.warning("Some inputs are outside the training domain. Prediction may be unreliable.")
        for w in warnings:
            st.write("-", w)

    tab1, tab2, tab3 = st.tabs(
        ["Cross sections", "Permittivity", "Data table"]
    )

    with tab1:
        st.pyplot(plot_cross_sections(df))

    with tab2:
        st.pyplot(plot_permittivity(df))

    with tab3:
        st.dataframe(df, use_container_width=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download prediction CSV",
            data=csv_bytes,
            file_name="predicted_cross_sections.csv",
            mime="text/csv",
        )

else:
    st.info("Choose parameters in the sidebar and click **Predict spectrum**.")
