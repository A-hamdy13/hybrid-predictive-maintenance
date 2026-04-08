"""Validate extracted rules: evidence, units, closed schema. Reject malformed entries."""
from typing import Optional

from config import ALLOWED_COMPONENTS, VALID_DISTANCE_UNITS, VALID_TIME_UNITS
from schemas import ExtractedManualRules, ExtractedRuleItem


def _validate_unit_distance(unit: Optional[str]) -> bool:
    return unit is None or (isinstance(unit, str) and unit.lower() in VALID_DISTANCE_UNITS)


def _validate_unit_time(unit: Optional[str]) -> bool:
    return unit is None or (isinstance(unit, str) and unit.lower() in VALID_TIME_UNITS)


def _validate_item(item: ExtractedRuleItem) -> tuple[bool, Optional[str]]:
    """Return (valid, error_message)."""
    if not item.found:
        return True, None

    if item.component not in ALLOWED_COMPONENTS:
        return False, f"Invalid component: {item.component}"

    # source_quote required only when there is no interval (scraped placeholder); user-entered rules may have interval only
    has_interval = item.interval_distance_value is not None or item.interval_time_value is not None
    if not has_interval and (not item.source_quote or not item.source_quote.strip()):
        return False, "Missing source_quote for found=true when no interval"

    if item.interval_distance_value is not None and not _validate_unit_distance(item.interval_distance_unit):
        return False, f"Invalid distance unit: {item.interval_distance_unit}"

    if item.interval_time_value is not None and not _validate_unit_time(item.interval_time_unit):
        return False, f"Invalid time unit: {item.interval_time_unit}"

    if item.confidence is not None and (item.confidence < 0 or item.confidence > 1):
        return False, "confidence must be 0-1"

    return True, None


def _has_actionable_interval(item: ExtractedRuleItem) -> bool:
    """True if the item has at least one concrete interval (distance or time).
    Items with found=True but both intervals null are not actionable for the decision engine."""
    if not item.found:
        return False
    return (
        item.interval_distance_value is not None
        or item.interval_time_value is not None
    )


def validate_extraction(rules: ExtractedManualRules) -> tuple[ExtractedManualRules, list[str]]:
    """Validate and filter rules. Returns (validated_rules, list of rejection messages).
    Drops items with found=True but no interval (distance and time both null) as non-actionable."""
    errors: list[str] = []
    valid_normal: list[ExtractedRuleItem] = []
    valid_severe: list[ExtractedRuleItem] = []

    for item in rules.service_schedule.normal_service:
        ok, err = _validate_item(item)
        if not ok:
            errors.append(f"normal_service {item.component}: {err}")
        elif _has_actionable_interval(item):
            valid_normal.append(item)
        else:
            errors.append(f"normal_service {item.component}: dropped (no interval)")

    for item in rules.service_schedule.severe_service:
        ok, err = _validate_item(item)
        if not ok:
            errors.append(f"severe_service {item.component}: {err}")
        elif _has_actionable_interval(item):
            valid_severe.append(item)
        else:
            errors.append(f"severe_service {item.component}: dropped (no interval)")

    validated = ExtractedManualRules(
        vehicle=rules.vehicle,
        service_schedule=rules.service_schedule.model_copy(
            update={"normal_service": valid_normal, "severe_service": valid_severe}
        ),
        source_url=rules.source_url,
        source_label=rules.source_label,
        source_urls=getattr(rules, "source_urls", None) or [],
    )
    return validated, errors


def verifier_pass(rules: ExtractedManualRules) -> list[str]:
    """Second pass: check that cited quote supports the extracted value/unit/component.
    Returns list of warnings (e.g. 'engine_oil: quote does not mention 10000 km'). 
    Does not remove entries; caller may use for logging.
    """
    warnings: list[str] = []
    for item in rules.service_schedule.normal_service + rules.service_schedule.severe_service:
        if not item.found or not item.source_quote:
            continue
        quote_lower = item.source_quote.lower()
        # Soft checks
        if item.interval_distance_value is not None and item.interval_distance_unit:
            val_str = str(int(item.interval_distance_value)) if item.interval_distance_value == int(item.interval_distance_value) else str(item.interval_distance_value)
            if val_str not in item.source_quote and str(item.interval_distance_value) not in item.source_quote:
                warnings.append(f"{item.component}: quote may not contain distance value {item.interval_distance_value} {item.interval_distance_unit}")
        if item.interval_time_value is not None and item.interval_time_unit:
            if str(int(item.interval_time_value)) not in item.source_quote and str(item.interval_time_value) not in item.source_quote:
                warnings.append(f"{item.component}: quote may not contain time value {item.interval_time_value} {item.interval_time_unit}")
    return warnings
