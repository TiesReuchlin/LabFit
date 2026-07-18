from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import uuid
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, abort, render_template, request, send_file
from scipy import odr

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2 MB

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------- Fit models ----------
def linear(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    a, b = beta
    return a * x + b


def exponential(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    a, b, c = beta
    return a * np.exp(b * x) + c


def power_law(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    a, b = beta
    return a * np.power(x, b)


MODELS: dict[str, dict] = {
    "linear": {
        "label": "Lineair: y = ax + b",
        "func": linear,
        "parameters": ["a", "b"],
    },
    "exponential": {
        "label": "Exponentieel: y = a·exp(bx) + c",
        "func": exponential,
        "parameters": ["a", "b", "c"],
    },
    "power": {
        "label": "Power law: y = a·xᵇ",
        "func": power_law,
        "parameters": ["a", "b"],
    },
}


def parse_number(value: object) -> float:
    """Accept decimal points and decimal commas."""
    if pd.isna(value):
        return math.nan
    text = str(value).strip().replace(",", ".")
    return float(text)


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "x": "x",
        "y": "y",
        "sigma_x": "sigma_x",
        "sigmax": "sigma_x",
        "sx": "sigma_x",
        "x_error": "sigma_x",
        "xerr": "sigma_x",
        "sigma_y": "sigma_y",
        "sigmay": "sigma_y",
        "sy": "sigma_y",
        "y_error": "sigma_y",
        "yerr": "sigma_y",
    }
    renamed = {}
    for column in df.columns:
        key = re.sub(r"[^a-z0-9_]+", "", str(column).strip().lower().replace("σ", "sigma_"))
        renamed[column] = aliases.get(key, key)
    df = df.rename(columns=renamed)

    if "x" not in df.columns or "y" not in df.columns:
        raise ValueError("De data moet minimaal kolommen met de namen x en y bevatten.")

    for optional in ("sigma_x", "sigma_y"):
        if optional not in df.columns:
            df[optional] = np.nan

    df = df[["x", "y", "sigma_x", "sigma_y"]].copy()
    for column in df.columns:
        df[column] = df[column].map(parse_number)

    df = df.dropna(subset=["x", "y"])
    if len(df) < 3:
        raise ValueError("Gebruik minimaal drie geldige meetpunten.")

    for column in ("sigma_x", "sigma_y"):
        invalid = df[column].notna() & (df[column] <= 0)
        if invalid.any():
            raise ValueError(f"Alle waarden in {column} moeten groter zijn dan nul.")

    return df


def read_input_data() -> pd.DataFrame:
    uploaded = request.files.get("csv_file")
    pasted = request.form.get("pasted_data", "").strip()

    if uploaded and uploaded.filename:
        raw = uploaded.read().decode("utf-8-sig")
    elif pasted:
        raw = pasted
    else:
        raise ValueError("Upload een CSV-bestand of plak meetgegevens in het tekstvak.")

    # Auto-detect comma, semicolon, tab, or whitespace-separated input.
    try:
        df = pd.read_csv(io.StringIO(raw), sep=None, engine="python")
    except Exception:
        df = pd.read_csv(io.StringIO(raw), sep=r"\s+", engine="python")

    return normalise_columns(df)


def initial_guess(model_name: str, x: np.ndarray, y: np.ndarray) -> list[float]:
    if model_name == "linear":
        slope, intercept = np.polyfit(x, y, 1)
        return [float(slope), float(intercept)]

    if model_name == "exponential":
        amplitude = float(y.max() - y.min()) or 1.0
        direction = 1.0 if y[-1] >= y[0] else -1.0
        scale = max(float(np.ptp(x)), 1e-9)
        return [amplitude, direction / scale, float(y.min())]

    # power law
    valid = (x > 0) & (y != 0)
    if valid.sum() >= 2 and np.all(y[valid] > 0):
        exponent, log_a = np.polyfit(np.log(x[valid]), np.log(y[valid]), 1)
        return [float(np.exp(log_a)), float(exponent)]
    return [1.0, 1.0]


def run_fit(df: pd.DataFrame, model_name: str) -> dict:
    if model_name not in MODELS:
        raise ValueError("Kies een geldig fitmodel.")

    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    sx = df["sigma_x"].to_numpy(dtype=float)
    sy = df["sigma_y"].to_numpy(dtype=float)

    if model_name == "power" and np.any(x <= 0):
        raise ValueError("Voor een power-law fit moeten alle x-waarden groter zijn dan nul.")

    model_info = MODELS[model_name]
    model = odr.Model(model_info["func"])

    sx_arg = sx if np.all(np.isfinite(sx)) else None
    sy_arg = sy if np.all(np.isfinite(sy)) else None
    data = odr.RealData(x, y, sx=sx_arg, sy=sy_arg)

    fit = odr.ODR(
        data,
        model,
        beta0=initial_guess(model_name, x, y),
        maxit=2000,
    ).run()

    if fit.info > 4 or not np.all(np.isfinite(fit.beta)):
        raise ValueError("De fit convergeerde niet. Controleer de data of probeer een ander model.")

    beta = fit.beta
    parameter_errors = fit.sd_beta
    y_pred = model_info["func"](beta, x)
    residuals = y - y_pred

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
    rmse = float(np.sqrt(np.mean(residuals**2)))

    order = np.argsort(x)
    x_sorted = x[order]
    span = float(np.ptp(x_sorted))
    padding = 0.04 * span if span > 0 else 1.0
    x_plot = np.linspace(x_sorted[0] - padding, x_sorted[-1] + padding, 500)
    if model_name == "power":
        x_plot = x_plot[x_plot > 0]
    y_plot = model_info["func"](beta, x_plot)

    parameters = []
    for name, value, error in zip(model_info["parameters"], beta, parameter_errors):
        parameters.append({
            "name": name,
            "value": float(value),
            "error": float(error),
            "formatted": f"{value:.6g} ± {error:.2g}",
        })

    equation = format_equation(model_name, beta)
    report_text = make_report_text(model_info["label"], parameters, r_squared, rmse)

    return {
        "x": x,
        "y": y,
        "sx": sx,
        "sy": sy,
        "y_pred": y_pred,
        "residuals": residuals,
        "x_plot": x_plot,
        "y_plot": y_plot,
        "parameters": parameters,
        "equation": equation,
        "r_squared": r_squared,
        "rmse": rmse,
        "report_text": report_text,
        "model_label": model_info["label"],
    }


def format_equation(model_name: str, beta: np.ndarray) -> str:
    if model_name == "linear":
        return f"y = ({beta[0]:.6g})x + ({beta[1]:.6g})"
    if model_name == "exponential":
        return f"y = ({beta[0]:.6g})·exp(({beta[1]:.6g})x) + ({beta[2]:.6g})"
    return f"y = ({beta[0]:.6g})·x^({beta[1]:.6g})"


def make_report_text(model_label: str, parameters: list[dict], r_squared: float, rmse: float) -> str:
    parameter_text = ", ".join(
        f"{item['name']} = {item['value']:.4g} ± {item['error']:.2g}"
        for item in parameters
    )
    r2_text = f"{r_squared:.4f}" if np.isfinite(r_squared) else "niet gedefinieerd"
    return (
        f"De meetgegevens zijn gefit met het model {model_label}. "
        f"De fitparameters zijn {parameter_text}. "
        f"De kwaliteit van de fit wordt beschreven door R² = {r2_text} "
        f"en RMSE = {rmse:.4g}. Controleer ook de residuen om te beoordelen "
        "of het gekozen model systematische afwijkingen vertoont."
    )


def make_plot(result: dict, x_label: str, y_label: str, title: str) -> bytes:
    fig, ax = plt.subplots(figsize=(8, 5.2))

    sx = result["sx"] if np.all(np.isfinite(result["sx"])) else None
    sy = result["sy"] if np.all(np.isfinite(result["sy"])) else None
    ax.errorbar(
        result["x"],
        result["y"],
        xerr=sx,
        yerr=sy,
        fmt="o",
        capsize=3,
        label="Meetdata",
    )
    ax.plot(result["x_plot"], result["y_plot"], label="Fit")
    ax.set_xlabel(x_label or "x")
    ax.set_ylabel(y_label or "y")
    ax.set_title(title or "LabFit-resultaat")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def save_result_files(result_id: str, result: dict, plot_png: bytes) -> None:
    (RESULTS_DIR / f"{result_id}.png").write_bytes(plot_png)

    output = pd.DataFrame({
        "x": result["x"],
        "y": result["y"],
        "y_fit": result["y_pred"],
        "residual": result["residuals"],
    })
    output.to_csv(RESULTS_DIR / f"{result_id}.csv", index=False)

    summary = {
        "model": result["model_label"],
        "equation": result["equation"],
        "parameters": result["parameters"],
        "r_squared": result["r_squared"],
        "rmse": result["rmse"],
        "report_text": result["report_text"],
    }
    (RESULTS_DIR / f"{result_id}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


@app.route("/", methods=["GET", "POST"])
def home():
    context = {
        "models": MODELS,
        "selected_model": "linear",
        "pasted_data": "x,y,sigma_x,sigma_y\n1,2.1,0.05,0.10\n2,4.0,0.05,0.12\n3,6.2,0.05,0.15\n4,8.1,0.05,0.16",
        "x_label": "x",
        "y_label": "y",
        "plot_title": "Meetdata met fit",
    }

    if request.method == "POST":
        context.update({
            "selected_model": request.form.get("model", "linear"),
            "pasted_data": request.form.get("pasted_data", ""),
            "x_label": request.form.get("x_label", "x"),
            "y_label": request.form.get("y_label", "y"),
            "plot_title": request.form.get("plot_title", "Meetdata met fit"),
        })
        try:
            df = read_input_data()
            result = run_fit(df, context["selected_model"])
            plot_png = make_plot(result, context["x_label"], context["y_label"], context["plot_title"])
            result_id = uuid.uuid4().hex
            save_result_files(result_id, result, plot_png)

            context.update({
                "result": result,
                "result_id": result_id,
                "plot_base64": base64.b64encode(plot_png).decode("ascii"),
            })
        except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as exc:
            context["error"] = str(exc)
        except Exception as exc:
            app.logger.exception("Unexpected fitting error")
            context["error"] = f"Onverwachte fout tijdens het fitten: {exc}"

    return render_template("index.html", **context)


@app.get("/download/<result_id>/<file_type>")
def download_result(result_id: str, file_type: str):
    if not re.fullmatch(r"[a-f0-9]{32}", result_id):
        abort(404)

    extensions = {"plot": "png", "data": "csv", "summary": "json"}
    extension = extensions.get(file_type)
    if not extension:
        abort(404)

    path = RESULTS_DIR / f"{result_id}.{extension}"
    if not path.exists():
        abort(404)

    return send_file(path, as_attachment=True, download_name=f"labfit_{file_type}.{extension}")


if __name__ == "__main__":
    app.run(debug=True)
