import logging
import numpy as np
from data_generation.parameters import *
from data_generation.operators import get_operator_profiles

def simulate_compression(drying_output, compression_pressure_setpoint, tablet_hardness_setpoint):
    """
    Simulates the compression process using externally injected control parameters.
    
    Parameters:
        drying_output (dict): Output from the drying stage containing (at minimum)
                              "status", "operator", and "moisture_content".
        compression_pressure_setpoint (float): Target compression pressure provided by the controller.
        tablet_hardness_setpoint (float): Target tablet hardness provided by the controller.
    
    Returns:
        dict: {
            "status": "success" or "failed",
            "operator": Operator name,
            "comp_pressure": Measured compression pressure,
            "tablet_hardness": Measured tablet hardness,
            "weight_variation": Measured weight variation,
            "dissolution": Computed dissolution value,
            "yield_percent": Computed yield percentage,
            "reason": Failure reason if any,
            "log": Detailed dictionary of intermediate values for feedback.
        }
    """
    log_details = {}
    
    if drying_output.get("status") != "success":
        return drying_output

    # Retrieve operator profile based on drying_output's operator
    operator_name = drying_output["operator"]
    operator = next(op for op in get_operator_profiles() if op.name == operator_name)
    log_details["operator"] = operator_name

    # Introduce machine drift
    machine_drift = np.random.normal(1.0, 0.01)
    log_details["machine_drift"] = machine_drift

    # Compute measured compression pressure using the injected setpoint, machine drift, and moisture effect.
    pressure_noise = np.random.normal(0, 0.5)
    measured_pressure = compression_pressure_setpoint * machine_drift * (1 + (np.exp(drying_output["moisture_content"] / 10) - 1)) + pressure_noise
    log_details["injected_compression_pressure_setpoint"] = compression_pressure_setpoint
    log_details["measured_pressure"] = measured_pressure

    # Compute measured tablet hardness using the injected setpoint and operator bias.
    tablet_hardness_noise = np.random.normal(0, 0.2)
    measured_tablet_hardness = tablet_hardness_setpoint * operator.compression_bias * machine_drift + tablet_hardness_noise
    log_details["injected_tablet_hardness_setpoint"] = tablet_hardness_setpoint
    log_details["measured_tablet_hardness"] = measured_tablet_hardness

    # Weight variation affected by moisture content (using a simplified formulation)
    weight_noise = np.random.normal(0, 0.1)
    weight_variation = (1.0 + drying_output["moisture_content"] / 5) + weight_noise
    log_details["weight_variation"] = weight_variation

    # Calculate dissolution and yield with added noise.
    dissolution_noise = np.random.normal(0, 1.0)
    dissolution = DISSOLUTION_BASE - (weight_variation * 2) - (drying_output["moisture_content"] * 1.5) + dissolution_noise
    yield_noise = np.random.normal(0, 1.5)
    yield_percent = YIELD_PERCENT_BASE - (weight_variation * 3) - (drying_output["moisture_content"] * 2) + yield_noise
    log_details["dissolution_noise"] = dissolution_noise
    log_details["yield_noise"] = yield_noise
    log_details["dissolution"] = dissolution
    log_details["yield_percent"] = yield_percent

    # Check for failure conditions.
    failures = []
    if dissolution < DISSOLUTION_MIN:
        failures.append(f"Dissolution too low ({dissolution:.1f} < {DISSOLUTION_MIN})")
    if yield_percent < YIELD_PERCENT_MIN:
        failures.append(f"Yield too low ({yield_percent:.1f}% < {YIELD_PERCENT_MIN}%)")

    if failures:
        result = {
            "status": "failed",
            "reason": ", ".join(failures),
            "operator": operator_name,
            "log": log_details
        }
        logging.debug(f"simulate_compression: Failing with result: {result}")
        return result

    result = {
        "status": "success",
        "operator": operator_name,
        "comp_pressure": measured_pressure,
        "tablet_hardness": measured_tablet_hardness,
        "weight_variation": weight_variation,
        "dissolution": dissolution,
        "yield_percent": yield_percent,
        "reason": "",
        "log": log_details
    }
    logging.debug(f"simulate_compression: Success with result: {result}")
    return result

if __name__ == "__main__":
    # Example test call:
    sample_drying_output = {
        "status": "success",
        "operator": "Alice",
        "moisture_content": 4.5  # example moisture content
    }
    test_result = simulate_compression(
        drying_output=sample_drying_output,
        compression_pressure_setpoint=(COMPRESSION_PRESSURE_MIN + COMPRESSION_PRESSURE_MAX) / 2.0,
        tablet_hardness_setpoint=(TABLET_HARDNESS_MIN + TABLET_HARDNESS_MAX) / 2.0
    )
    print(test_result)
