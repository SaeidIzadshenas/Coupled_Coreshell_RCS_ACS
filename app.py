import os
import json
import yaml
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# =========================
# Files
# =========================
MODEL_FILE = "best_model.keras"
X_SCALER_FILE = "x_scaler.pkl"
Y_SCALER_FILE = "y_scaler.pkl"
DOMAIN_FILE = "training_domain.json"

RI_DB = "refractiveindex.info-database/database/data/main"

INPUT_COLS = [
    "r0", "d", "g", "n0", "lda0",
    "eps_c1", "eps_c2", "eps_s1", "eps_s2"
]

OUTPUT_COLS = [
    "C_scatt_total",
    "C_absor_total",
    "C_ext_total"
]

# =========================
# Load model and scalers
# =========================
@st.cache_resource
def load_ai_model():
    model = load_model(MODEL_FILE)
    x_scaler = joblib.load(X_SCALER_FILE)
    y_scaler = joblib.load(Y_SCALER_FILE)

    with open(DOMAIN_FILE, "r") as f:
        domain = json.load(f)

    return model, x_scaler, y_scaler, domain


# =========================
# Find materials
# =========================
@st.cache_data
def find_material_files():
    materials = []

    for root, dirs, files in os.walk(RI_DB):
        for file in files:
            if file.endswith(".yml"):
                path = os.path.join(root, file)

                try:
                    with open(path, "r", encoding="utf-8") as f:
                        txt = f.read()

                    if "tabulated nk" in txt:
                        rel = os.path.relpath(path, RI_DB)
                        materials.append(rel)

                except:
                    pass

    return sorted(materials)


# =========================
# Read n,k from YAML
# =========================
@st.cache_data
def read_nk(material_relative_path):
    path = os.path.join(RI_DB, material_relative_path)

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    blocks = data.get("DATA", [])

    for block in blocks:
        if "tabulated nk" in block.get("type", ""):
            raw = block["data"].strip().splitlines()

            wl_um = []
            n_list = []
            k_list = []

            for line in raw:
                parts = line.split()
                if len(parts) >= 3:
                    wl_um.append(float(parts[0]))
                    n_list.append(float(parts[1]))
                    k_list.append(float(parts[2]))

            wl_nm = np.array(wl_um) * 1000.0
            n_arr = np.array(n_list)
            k_arr = np.array(k_list)

            return wl_nm, n_arr, k_arr

    raise ValueError("No tabulated nk data found.")


def interpolate_nk(material, wavelengths_nm):
    wl_nm, n_arr, k_arr = read_nk(material)

    wl_min = wl_nm.min()
    wl_max = wl_nm.max()

    if wavelengths_nm.min() < wl_min or wavelengths_nm.max() > wl_max:
        st.warning(
            f"Material data warning for {material}: "
            f"available wavelength range is {wl_min:.1f}–{wl_max:.1f} nm."
        )

    n_interp = np.interp(wavelengths_nm, wl_nm, n_arr)
    k_interp = np.interp(wavelengths_nm, wl_nm, k_arr)

    return n_interp, k_interp


# =========================
# Check training domain
# =========================
def check_domain(X, domain):
    warnings = []

    for j, col in enumerate(INPUT_COLS):
        qmin = np.min(X[:, j])
        qmax = np.max(X[:, j])

        dmin = domain[col]["min"]
        dmax = domain[col]["max"]

        if qmin < dmin or qmax > dmax:
            warnings.append(
                f"{col}: selected range [{qmin:.4g}, {qmax:.4g}] "
                f"is outside training range [{dmin:.4g}, {dmax:.4g}]"
            )

    return warnings


# =========================
# Prediction function
# =========================
def predict_spectra(
    model, x_scaler, y_scaler, domain,
    r0, d, g, n0,
    wl_start, wl_stop, wl_points,
    core_material, shell_material
):
    wavelengths = np.linspace(wl_start, wl_stop, wl_points)

    n_core, k_core = interpolate_nk(core_material, wavelengths)
    n_shell, k_shell = interpolate_nk(shell_material, wavelengths)

    eps_core = (n_core + 1j * k_core) ** 2
    eps_shell = (n_shell + 1j * k_shell) ** 2

    X = np.column_stack([
        np.full_like(wavelengths, r0),
        np.full_like(wavelengths, d),
        np.full_like(wavelengths, g),
        np.full_like(wavelengths, n0),
        wavelengths,
        eps_core.real,
        eps_core.imag,
        eps_shell.real,
        eps_shell.imag
    ])

    domain_warnings = check_domain(X, domain)

    X_scaled = x_scaler.transform(X)

    y_scaled = model.predict(X_scaled, verbose=0)

    y_log10 = y_scaler.inverse_transform(y_scaled)

    y_real = 10 ** y_log10
    y_real = np.maximum(y_real, 0)

    df = pd.DataFrame({
        "wavelength_nm": wavelengths,
        "C_scatt_total": y_real[:, 0],
        "C_absor_total": y_real[:, 1],
        "C_ext_total_model": y_real[:, 2],
    })

    df["C_ext_total_sum"] = df["C_scatt_total"] + df["C_absor_total"]

    return df, domain_warnings


# =========================
# Streamlit app
# =========================
st.set_page_config(page_title="Nanophotonics Spectrum Predictor", layout="wide")

st.title("Local Nanophotonics Spectrum Predictor")

st.write(
    "This app uses your trained neural-network model and refractiveindex.info material data "
    "to predict scattering, absorption, and extinction spectra."
)

model, x_scaler, y_scaler, domain = load_ai_model()
materials = find_material_files()

if len(materials) == 0:
    st.error("No material files found. Check refractiveindex.info database folder.")
    st.stop()

st.sidebar.header("Design parameters")

r0 = st.sidebar.slider("Core radius r0 (nm)", 5.0, 100.0, 75.0)
d = st.sidebar.slider("Shell thickness d (nm)", 1.0, 50.0, 10.0)
g = st.sidebar.slider("Gap g (nm)", 2.0, 50.0, 10.0)
n0 = st.sidebar.slider("Background refractive index n0", 1.0, 1.52, 1.35)

st.sidebar.header("Wavelength range")

wl_start = st.sidebar.number_input("Start wavelength (nm)", value=400.0)
wl_stop = st.sidebar.number_input("Stop wavelength (nm)", value=900.0)
wl_points = st.sidebar.slider("Number of wavelength points", 50, 1000, 300)

st.sidebar.header("Materials")

core_material = st.sidebar.selectbox(
    "Core material",
    materials,
    index=0
)

shell_material = st.sidebar.selectbox(
    "Shell material",
    materials,
    index=0
)

run_button = st.sidebar.button("Predict spectra")

if run_button:
    df, warnings = predict_spectra(
        model, x_scaler, y_scaler, domain,
        r0, d, g, n0,
        wl_start, wl_stop, wl_points,
        core_material, shell_material
    )

    if warnings:
        st.warning("Some inputs are outside the training domain. Prediction may be unreliable.")
        for w in warnings:
            st.write("-", w)

    st.subheader("Predicted spectra")

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(df["wavelength_nm"], df["C_scatt_total"], label="C_scatt_total")
    ax.plot(df["wavelength_nm"], df["C_absor_total"], label="C_absor_total")
    ax.plot(df["wavelength_nm"], df["C_ext_total_model"], label="C_ext_total model")
    ax.plot(df["wavelength_nm"], df["C_ext_total_sum"], "--", label="C_scatt + C_absor")

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Cross section")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    st.pyplot(fig)

    st.subheader("Data table")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download predicted spectra as CSV",
        data=csv,
        file_name="predicted_spectra.csv",
        mime="text/csv"
    )

else:
    st.info("Select parameters from the left side, then click Predict spectra.")