"""Combine rule-based schedule status and optional ML output into final recommendation."""
from typing import Optional

from schemas import (
    ComponentStatus,
    DecisionOutput,
    ExtractedManualRules,
    MLOutput,
    OperationalInputs,
)


def _km_or_miles(value: Optional[float], unit: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    if unit and "mile" in unit.lower():
        return value * 1.60934  # to km for internal comparison
    return value


def _evaluate_component(
    component: str,
    rule_item: dict,
    inputs: OperationalInputs,
) -> tuple[str, str, str]:  # status, priority, reason
    """Evaluate one rule item against user inputs. Returns (status, priority, reason)."""
    interval_km = _km_or_miles(
        rule_item.get("interval_distance_value"),
        rule_item.get("interval_distance_unit"),
    )
    interval_months = None
    if rule_item.get("interval_time_unit") == "months" and rule_item.get("interval_time_value") is not None:
        interval_months = rule_item.get("interval_time_value")

    current_km = inputs.current_mileage_km
    reason_parts = []
    interval_used = []

    # Distance-based
    if interval_km and interval_km > 0:
        interval_used.append(f"every {interval_km:.0f} km")
        if component == "engine_oil" or component == "oil_filter":
            since = inputs.mileage_since_last_oil_change or (current_km - 0)
            if since is not None:
                pct = (since / interval_km) * 100
                if pct >= 100:
                    return "overdue", "red", f"Overdue by distance ({since:.0f} km since last service; interval {interval_km:.0f} km)."
                if pct >= 80:
                    return "due_soon", "yellow", f"Due soon by distance ({since:.0f} km / {interval_km:.0f} km)."
                reason_parts.append(f"{since:.0f} km / {interval_km:.0f} km")
        elif component == "brake_inspection":
            since = inputs.mileage_since_brake_service
            if since is not None and current_km is not None:
                since_km = since
                pct = (since_km / interval_km) * 100
                if pct >= 100:
                    return "overdue", "red", f"Brake inspection overdue ({since_km:.0f} km / {interval_km:.0f} km)."
                if pct >= 80:
                    return "due_soon", "yellow", f"Brake inspection due soon ({since_km:.0f} km / {interval_km:.0f} km)."
                reason_parts.append(f"{since_km:.0f} km / {interval_km:.0f} km")
        elif component == "tire_rotation":
            since = inputs.mileage_since_tire_rotation
            if since is not None:
                pct = (since / interval_km) * 100
                if pct >= 100:
                    return "overdue", "red", f"Tire rotation overdue ({since:.0f} km / {interval_km:.0f} km)."
                if pct >= 80:
                    return "due_soon", "yellow", f"Tire rotation due soon ({since:.0f} km / {interval_km:.0f} km)."
                reason_parts.append(f"{since:.0f} km / {interval_km:.0f} km")

    # Time-based (e.g. oil every 6 months)
    if interval_months and interval_months > 0:
        interval_used.append(f"every {interval_months:.0f} months")
        if component in ("engine_oil", "oil_filter") and inputs.months_since_last_oil_change is not None:
            m = inputs.months_since_last_oil_change
            if m >= interval_months:
                return "overdue", "red", f"Overdue by time ({m:.0f} months since last oil change; interval {interval_months:.0f} months)."
            if m >= interval_months * 0.8:
                return "due_soon", "yellow", f"Due soon by time ({m:.0f} / {interval_months:.0f} months)."
            reason_parts.append(f"{m:.0f} / {interval_months:.0f} months")

    if reason_parts:
        return "ok", "green", f"Within interval ({', '.join(reason_parts)})."
    return "ok", "green", f"Scheduled: {', '.join(interval_used) or 'see manual'}."


def run_decision_engine(
    rules: ExtractedManualRules | None,
    inputs: OperationalInputs,
    ml_output: Optional[MLOutput] = None,
    use_severe_service: bool = False,
) -> DecisionOutput:
    """Combine rules and optional ML into final recommendation.
    use_severe_service: if True, evaluate against severe_service intervals; else normal_service."""
    components: list[ComponentStatus] = []
    priorities = {"green": 0, "yellow": 1, "red": 2}
    max_priority = "green"

    if rules:
        schedule = rules.service_schedule.severe_service if use_severe_service else rules.service_schedule.normal_service
        if not schedule:
            schedule_name = "severe service" if use_severe_service else "normal service"
            components.append(
                ComponentStatus(
                    component="overall",
                    status="no_rule",
                    priority="green",
                    reason=f"No {schedule_name} rules for this vehicle. Add rules or use the other schedule.",
                    interval_used=None,
                )
            )
        for item in schedule:
            if not item.found:
                components.append(
                    ComponentStatus(
                        component=item.component,
                        status="no_rule",
                        priority="green",
                        reason="No interval in manual.",
                        interval_used=None,
                    )
                )
                continue
            status, priority, reason = _evaluate_component(
                item.component,
                item.model_dump(),
                inputs,
            )
            interval_used = None
            if item.interval_distance_value is not None and item.interval_distance_unit:
                interval_used = f"{item.interval_distance_value:.0f} {item.interval_distance_unit}"
            if item.interval_time_value is not None and item.interval_time_unit:
                interval_used = (interval_used or "") + (f" / {item.interval_time_value:.0f} {item.interval_time_unit}" if interval_used else f"{item.interval_time_value:.0f} {item.interval_time_unit}")

            components.append(
                ComponentStatus(
                    component=item.component,
                    status=status,
                    priority=priority,
                    reason=reason,
                    interval_used=interval_used,
                )
            )
            if priorities[priority] > priorities[max_priority]:
                max_priority = priority

        # Rule–ML fusion: manual is stronger; ML can upgrade or add "inspect soon"
        # - Manual overdue → keep red
        # - Manual due_soon + ML risk high → high priority (red)
        # - Manual ok + ML risk high → inspect soon (yellow)
        # - Manual missing + ML risk high → add component "inspect soon"
        # - ML low + manual ok → keep green
        if ml_output and ml_output.enabled and ml_output.component_risks:
            comp_risks = ml_output.component_risks
            rule_component_names = {c.component for c in components}
            new_components: list[ComponentStatus] = []
            for c in components:
                risk = comp_risks.get(c.component)
                ml_pct = f" Telemetry risk: {risk * 100:.0f}%." if risk is not None else ""

                if c.priority == "red":
                    # Manual overdue → keep red; append ML % for context
                    new_components.append(
                        ComponentStatus(
                            component=c.component,
                            status=c.status,
                            priority="red",
                            reason=c.reason.rstrip(".") + ml_pct if ml_pct else c.reason,
                            interval_used=c.interval_used,
                        )
                    )
                    max_priority = "red"
                elif c.priority == "yellow":
                    # Due soon + ML risk high → high priority (red); else keep yellow
                    if risk is not None and risk >= 0.5:
                        new_components.append(
                            ComponentStatus(
                                component=c.component,
                                status=c.status,
                                priority="red",
                                reason=c.reason.rstrip(".") + f"; telemetry suggests elevated risk ({risk * 100:.0f}%). Schedule and sensors both warrant priority.",
                                interval_used=c.interval_used,
                            )
                        )
                        max_priority = "red"
                    else:
                        new_components.append(
                            ComponentStatus(
                                component=c.component,
                                status=c.status,
                                priority="yellow",
                                reason=c.reason.rstrip(".") + ml_pct if ml_pct else c.reason,
                                interval_used=c.interval_used,
                            )
                        )
                        if priorities["yellow"] > priorities[max_priority]:
                            max_priority = "yellow"
                else:
                    # Green (ok): ML high → inspect soon; ML low → keep green
                    if risk is not None and risk >= 0.5:
                        new_components.append(
                            ComponentStatus(
                                component=c.component,
                                status="inspect_recommended",
                                priority="yellow",
                                reason=c.reason.rstrip(".") + f" Telemetry suggests elevated risk ({risk * 100:.0f}%); inspect soon.",
                                interval_used=c.interval_used,
                            )
                        )
                        if priorities["yellow"] > priorities[max_priority]:
                            max_priority = "yellow"
                    else:
                        new_components.append(
                            ComponentStatus(
                                component=c.component,
                                status=c.status,
                                priority="green",
                                reason=c.reason.rstrip(".") + ml_pct if ml_pct else c.reason,
                                interval_used=c.interval_used,
                            )
                        )
                        if priorities[c.priority] > priorities[max_priority]:
                            max_priority = c.priority
            components = new_components

            # Add ML-only components (no manual rule): telemetry suggests risk → inspect soon
            for comp_name, risk in comp_risks.items():
                if comp_name == "overall" or comp_name in rule_component_names:
                    continue
                if risk is not None and risk >= 0.5:
                    components.append(
                        ComponentStatus(
                            component=comp_name,
                            status="inspect_recommended",
                            priority="yellow",
                            reason=f"No schedule rule; telemetry suggests elevated risk ({risk * 100:.0f}%). Inspect soon.",
                            interval_used=None,
                        )
                    )
                    if priorities["yellow"] > priorities[max_priority]:
                        max_priority = "yellow"
    else:
        # No rules: if ML only, still produce a summary
        if ml_output and ml_output.enabled:
            components.append(
                ComponentStatus(
                    component="overall",
                    status="inspect_recommended",
                    priority="yellow" if (ml_output.overall_risk_score or 0) > 0.5 else "green",
                    reason=f"ML risk score: {(ml_output.overall_risk_score or 0):.2f}. No manual rules loaded.",
                    interval_used=None,
                )
            )
            max_priority = "yellow" if ((ml_output.overall_risk_score or 0) > 0.5) else "green"
        else:
            components.append(
                ComponentStatus(
                    component="overall",
                    status="no_rule",
                    priority="green",
                    reason="No manual rules loaded and ML disabled. Add a vehicle and extract rules first.",
                    interval_used=None,
                )
            )

    # Summary sentence
    red_count = sum(1 for c in components if c.priority == "red")
    yellow_count = sum(1 for c in components if c.priority == "yellow")
    if red_count > 0:
        summary = f"{red_count} item(s) overdue; schedule maintenance soon."
    elif yellow_count > 0:
        summary = f"{yellow_count} item(s) due soon or elevated risk; consider scheduling."
    else:
        summary = "All evaluated items within schedule. Continue routine monitoring."

    return DecisionOutput(
        components=components,
        overall_priority=max_priority,
        summary=summary,
        ml_output=ml_output,
        rule_engine_only=not (ml_output and ml_output.enabled),
    )
