"""Telemetry-based ML prediction (optional). When disabled, UI uses rules only."""
import logging
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from config import BASE_DIR
from schemas import MLOutput, OperationalInputs

log = logging.getLogger(__name__)

MODEL_PATH = BASE_DIR / "data" / "model.pkl"
FEATURES_PATH = BASE_DIR / "data" / "model_features.txt"
COMPONENTS_PATH = BASE_DIR / "data" / "model_components.pkl"

# Aliases: OperationalInputs may send these under alternate names (UI sends both).
FEATURE_ALIASES: dict[str, str] = {
    "odometer_reading": "current_mileage_km",
    "engine_temp_c": "engine_temperature_c",
}

_model_cache: Optional[Any] = None
_feature_order: Optional[list[str]] = None
_component_models_cache: Optional[dict[str, Any]] = None


def _load_model():
    global _model_cache, _feature_order, _component_models_cache
    if _model_cache is not None and _feature_order is not None:
        return _model_cache, _feature_order

    if not MODEL_PATH.exists():
        log.debug("[ml_predictor] No model.pkl at %s; using placeholder", MODEL_PATH)
        return None, None

    try:
        import joblib
    except ImportError:
        log.warning("[ml_predictor] joblib not installed; cannot load model")
        return None, None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            _model_cache = joblib.load(MODEL_PATH)
        _feature_order = []
        if FEATURES_PATH.exists():
            with open(FEATURES_PATH, encoding="utf-8") as f:
                _feature_order = [line.strip() for line in f if line.strip()]
        if not _feature_order:
            log.warning("[ml_predictor] No feature list; cannot build input vector")
            _model_cache = None
            return None, None

        if COMPONENTS_PATH.exists():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                _component_models_cache = joblib.load(COMPONENTS_PATH)
        else:
            _component_models_cache = None

        log.info("[ml_predictor] Loaded model from %s (%s features)", MODEL_PATH, len(_feature_order))
        return _model_cache, _feature_order
    except Exception as e:
        log.warning("[ml_predictor] Failed to load model: %s", e)
        return None, None


def _get_feature_value(inputs: OperationalInputs, feature_name: str) -> Optional[float]:
    """Get feature value from inputs; use alias if the primary attribute is missing."""
    v = getattr(inputs, feature_name, None)
    if v is not None:
        return float(v)
    alt = FEATURE_ALIASES.get(feature_name)
    if alt:
        v = getattr(inputs, alt, None)
        if v is not None:
            return float(v)
    return None


def _inputs_to_feature_row(inputs: OperationalInputs, feature_order: list[str]) -> Optional[pd.DataFrame]:
    """Build one row from user-provided inputs only. Returns None if any feature is missing.
    Returns a DataFrame with feature names so sklearn does not warn about missing feature names."""
    values: list[float] = []
    for name in feature_order:
        v = _get_feature_value(inputs, name)
        if v is None:
            return None
        values.append(v)
    return pd.DataFrame([values], columns=feature_order)


def get_ml_prediction(inputs: OperationalInputs, *, enable: bool = True) -> MLOutput:
    """Return ML risk scores when enabled. Uses saved model if available, else placeholder."""
    if not enable:
        return MLOutput(enabled=False)

    model, feature_order = _load_model()
    if model is None or feature_order is None:
        return _placeholder_prediction(inputs)

    try:
        X = _inputs_to_feature_row(inputs, feature_order)
        if X is None:
            log.debug("[ml_predictor] Incomplete telemetry inputs; using placeholder")
            return _placeholder_prediction(inputs)
        proba = getattr(model, "predict_proba", None)
        if proba is not None:
            risk = float(proba(X)[0, 1])
        else:
            pred = model.predict(X)[0]
            risk = float(pred)

        component_risks = {"overall": risk}
        comp_key = {"engine": "engine_oil", "brake": "brake_inspection", "battery": "battery"}
        if _component_models_cache:
            for comp_name, comp_model in _component_models_cache.items():
                if hasattr(comp_model, "predict_proba"):
                    p = comp_model.predict_proba(X)[0, 1]
                    component_risks[comp_key.get(comp_name, comp_name)] = float(p)

        return MLOutput(
            enabled=True,
            overall_risk_score=min(risk, 1.0),
            failure_probability=risk * 0.9,
            component_risks=component_risks,
            anomaly_indicator=risk > 0.5,
        )
    except Exception as e:
        log.warning("[ml_predictor] Prediction failed: %s; falling back to placeholder", e)
        return _placeholder_prediction(inputs)


def _placeholder_prediction(inputs: OperationalInputs) -> MLOutput:
    """Heuristic fallback when no trained model is loaded."""
    risk = 0.0
    component_risks: dict[str, float] = {}

    if inputs.engine_temperature_c is not None:
        if inputs.engine_temperature_c > 105:
            component_risks["engine_oil"] = 0.8
            risk = max(risk, 0.7)
        elif inputs.engine_temperature_c > 98:
            component_risks["engine_oil"] = 0.4
            risk = max(risk, 0.3)
    if inputs.oil_pressure_psi is not None and inputs.oil_pressure_psi < 25:
        component_risks["engine_oil"] = max(component_risks.get("engine_oil", 0), 0.7)
        risk = max(risk, 0.6)
    if inputs.battery_voltage_v is not None:
        if inputs.battery_voltage_v < 11.8:
            component_risks["battery"] = 0.8
            risk = max(risk, 0.6)
        elif inputs.battery_voltage_v < 12.2:
            component_risks["battery"] = 0.4
            risk = max(risk, 0.3)
    if inputs.tire_pressure_psi is not None and inputs.tire_pressure_psi < 28:
        component_risks["tire_rotation"] = 0.5
        risk = max(risk, 0.35)

    if not component_risks:
        component_risks = {"overall": 0.2}

    return MLOutput(
        enabled=True,
        overall_risk_score=min(risk + 0.1, 1.0),
        failure_probability=risk * 0.9,
        component_risks=component_risks,
        anomaly_indicator=risk > 0.5,
    )
