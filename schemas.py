"""Pydantic schemas for rules, API, and decision engine."""
from typing import Literal, Optional

from pydantic import BaseModel, Field


# --- Vehicle identity ---
class VehicleIdentity(BaseModel):
    make: str
    model: str
    year: int
    trim_or_engine: Optional[str] = None


# --- Registered vehicle (per license plate) ---
class VehicleRecord(BaseModel):
    license_plate: str
    make: str
    model: str
    year: int
    trim_or_engine: Optional[str] = None
    notes: Optional[str] = None
    severe_service_flags: Optional[dict[str, bool]] = None  # condition id -> true if vehicle meets it (use severe schedule)

    def to_identity(self) -> "VehicleIdentity":
        return VehicleIdentity(
            make=self.make,
            model=self.model,
            year=self.year,
            trim_or_engine=self.trim_or_engine,
        )


# --- Manual retrieval ---
class ManualCandidate(BaseModel):
    url: str
    title: str
    source_type: Literal["owner_manual", "maintenance_guide", "service_manual", "other"]
    is_official: bool = True


# --- Extracted rule (single service item) ---
class ExtractedRuleItem(BaseModel):
    component: str
    action: str  # replace, inspect, check, etc.
    interval_distance_value: Optional[float] = None
    interval_distance_unit: Optional[str] = None
    interval_time_value: Optional[float] = None
    interval_time_unit: Optional[str] = None
    conditions: Optional[str] = None
    found: bool = True
    source_page: Optional[int] = None
    source_quote: Optional[str] = None
    source_url: Optional[str] = None  # URL this rule was extracted from
    confidence: Optional[float] = None


class ServiceSchedule(BaseModel):
    normal_service: list[ExtractedRuleItem] = []
    severe_service: list[ExtractedRuleItem] = []


class SevereServiceCondition(BaseModel):
    """One condition that, if met, means the vehicle is under 'severe service' (use severe schedule)."""
    id: str  # slug, e.g. short_trips, towing
    label: str
    description: Optional[str] = None


class ExtractedManualRules(BaseModel):
    vehicle: VehicleIdentity
    service_schedule: ServiceSchedule
    source_url: Optional[str] = None
    source_label: Optional[str] = None
    source_urls: list[str] = []  # all URLs that contributed to this rule set (when merged from multiple)
    severe_service_conditions: list[SevereServiceCondition] = []  # scraped: conditions that qualify as severe service


# --- User location (optional, for nearby mechanics) ---
class UserLocation(BaseModel):
    latitude: float = Field(..., description="Latitude in decimal degrees")
    longitude: float = Field(..., description="Longitude in decimal degrees")
    accuracy_m: Optional[float] = Field(
        default=None,
        description="Optional accuracy radius in meters if provided by the browser/device",
    )


class NearbyMechanic(BaseModel):
    kind: Literal["affiliate", "general", "dealer", "brake_tire"]
    label: str
    name: str
    maps_url: str


# --- User operational inputs (telemetry / maintenance) ---
# Fields below match model feature names where used for ML; aliases kept for UI (e.g. current_mileage_km → odometer_reading).
class OperationalInputs(BaseModel):
    current_mileage_km: Optional[float] = None
    months_since_last_oil_change: Optional[float] = None
    mileage_since_last_oil_change: Optional[float] = None
    mileage_since_brake_service: Optional[float] = None
    mileage_since_tire_rotation: Optional[float] = None
    battery_age_months: Optional[float] = None
    vehicle_age_months: Optional[float] = None
    # Telemetry (shared / basic)
    engine_rpm: Optional[float] = None
    engine_temperature_c: Optional[float] = None
    oil_pressure_psi: Optional[float] = None
    battery_voltage_v: Optional[float] = None
    tire_pressure_psi: Optional[float] = None
    fuel_consumption_lph: Optional[float] = None
    # ML model features (same names as model_features.txt; shown in UI when Use ML is on)
    odometer_reading: Optional[float] = None
    engine_temp_c: Optional[float] = None
    coolant_temp_c: Optional[float] = None
    fuel_level_percent: Optional[float] = None
    engine_load_percent: Optional[float] = None
    throttle_pos_percent: Optional[float] = None
    air_flow_rate_gps: Optional[float] = None
    exhaust_gas_temp_c: Optional[float] = None
    vibration_level: Optional[float] = None
    engine_hours: Optional[float] = None
    brake_fluid_level_psi: Optional[float] = None
    brake_pad_wear_mm: Optional[float] = None
    brake_temp_c: Optional[float] = None
    abs_fault_indicator: Optional[float] = None
    brake_pedal_pos_percent: Optional[float] = None
    wheel_speed_fl_kph: Optional[float] = None
    wheel_speed_fr_kph: Optional[float] = None
    wheel_speed_rl_kph: Optional[float] = None
    wheel_speed_rr_kph: Optional[float] = None
    battery_current_a: Optional[float] = None
    battery_temp_c: Optional[float] = None
    alternator_output_v: Optional[float] = None
    battery_charge_percent: Optional[float] = None
    battery_health_percent: Optional[float] = None
    vehicle_speed_kph: Optional[float] = None
    ambient_temp_c: Optional[float] = None
    humidity_percent: Optional[float] = None


# --- Rule engine output (per component) ---
class ComponentStatus(BaseModel):
    component: str
    status: Literal["ok", "due_soon", "overdue", "no_rule", "inspect_recommended"]
    priority: Literal["green", "yellow", "red"]
    reason: str
    interval_used: Optional[str] = None


# --- ML output (when enabled) ---
class MLOutput(BaseModel):
    enabled: bool = False
    overall_risk_score: Optional[float] = None
    failure_probability: Optional[float] = None
    component_risks: Optional[dict[str, float]] = None
    anomaly_indicator: Optional[bool] = None


# --- Decision engine output ---
class DecisionOutput(BaseModel):
    components: list[ComponentStatus] = []
    overall_priority: Literal["green", "yellow", "red"]
    summary: str
    ml_output: Optional[MLOutput] = None
    rule_engine_only: bool = False


# --- Per-vehicle log entries ---
class LogRecommendationEntry(BaseModel):
    at: str  # ISO timestamp
    decision: dict  # DecisionOutput.model_dump()
    explanation: str
    inputs_snapshot: dict  # OperationalInputs.model_dump()
    use_severe_service: bool = False


class LogAlertEntry(BaseModel):
    at: str
    kind: str  # e.g. "overdue", "due_soon"
    message: str
    component: Optional[str] = None


class VehicleLog(BaseModel):
    license_plate: str
    recommendations: list[LogRecommendationEntry] = []
    operational_inputs: list[dict] = []  # history of inputs entered for this vehicle
    alerts: list[LogAlertEntry] = []


# --- API request/response ---
class RecommendationRequest(BaseModel):
    vehicle: VehicleIdentity
    inputs: OperationalInputs
    use_ml: bool = False
    use_severe_service: bool = False
    ui_language: Literal["en", "ar", "hi"] = "en"
    user_location: Optional[UserLocation] = None
    license_plate: Optional[str] = None  # if set, recommendation is logged for this vehicle


class RecommendationResponse(BaseModel):
    decision: DecisionOutput
    explanation: str
    vehicle: VehicleIdentity
    nearby_mechanics: Optional[list[NearbyMechanic]] = None
