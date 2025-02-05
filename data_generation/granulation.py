import logging
import numpy as np
from data_generation.parameters import *

def simulate_granulation(mixing_output, granulation_time_setpoint, binder_rate_setpoint):
    """
    Simulates the granulation process using externally injected control parameters.
    
    Parameters:
        mixing_output (dict): Output from the mixing stage containing, at minimum, the 'status' and 'uniformity_index'.
        granulation_time_setpoint (float): The target granulation time provided by the controller.
        binder_rate_setpoint (float): The target binder rate provided by the controller.
    
    Returns:
        dict: A dictionary containing:
            - "status": "success" or "failed"
            - "operator": Operator identifier (passed from mixing stage)
            - "granulation_time": The measured granulation time (after drift/noise)
            - "binder_rate": The measured binder rate (after drift/noise)
            - "granule_density": The computed granule density
            - "reason": Failure reason if applicable
            - "log": A dictionary with intermediate values and feedback details
    """
    log_details = {}

    # Check if previous stage succeeded; if not, propagate failure.
    if mixing_output.get("status") != "success":
        return mixing_output

    operator = mixing_output["operator"]
    log_details["operator"] = operator
    log_details["mixing_uniformity_index"] = mixing_output.get("uniformity_index")

    # Introduce machine drift and noise
    machine_drift = np.random.normal(1.0, 0.01)
    log_details["machine_drift"] = machine_drift

    granulation_time_noise = np.random.normal(0, 0.3)
    binder_rate_noise = np.random.normal(0, 0.3)

    # Compute the actual (measured) granulation time and binder rate using the injected setpoints.
    measured_granulation_time = granulation_time_setpoint * machine_drift + granulation_time_noise
    measured_binder_rate = binder_rate_setpoint * machine_drift + binder_rate_noise

    log_details["injected_granulation_time_setpoint"] = granulation_time_setpoint
    log_details["injected_binder_rate_setpoint"] = binder_rate_setpoint
    log_details["measured_granulation_time"] = measured_granulation_time
    log_details["measured_binder_rate"] = measured_binder_rate

    # Define ideal values for reference (midpoints of allowed ranges).
    ideal_granulation_time = (GRANULATION_TIME_MIN + GRANULATION_TIME_MAX) / 2.0
    ideal_binder_rate = (BINDER_RATE_MIN + BINDER_RATE_MAX) / 2.0

    log_details["ideal_granulation_time"] = ideal_granulation_time
    log_details["ideal_binder_rate"] = ideal_binder_rate

    # Compute deviations from ideal targets.
    time_deviation = abs(measured_granulation_time - ideal_granulation_time) / ideal_granulation_time
    binder_deviation = abs(measured_binder_rate - ideal_binder_rate) / ideal_binder_rate
    log_details["time_deviation"] = time_deviation
    log_details["binder_deviation"] = binder_deviation

    # Compute granule density influenced by the mixing uniformity and the measured binder rate.
    granule_density = (mixing_output["uniformity_index"] * 0.8) + (measured_binder_rate * 0.05) + np.random.normal(0, 0.05)
    log_details["granule_density"] = granule_density

    # Check if granule density falls below the required threshold.
    if granule_density < GRANULE_DENSITY_MIN:
        result = {
            "status": "failed",
            "reason": f"Granule density too low ({granule_density:.2f} < {GRANULE_DENSITY_MIN})",
            "operator": operator,
            "log": log_details
        }
        logging.debug(f"simulate_granulation: Failing with result: {result}")
        return result

    result = {
        "status": "success",
        "operator": operator,
        "granulation_time": measured_granulation_time,
        "binder_rate": measured_binder_rate,
        "granule_density": granule_density,
        "reason": "",
        "log": log_details
    }

    logging.debug(f"simulate_granulation: Success with result: {result}")
    return result

if __name__ == "__main__":
    # Example test call:
    # Assume a successful mixing output with a given uniformity index.
    sample_mixing_output = {
        "status": "success",
        "operator": "Alice",
        "uniformity_index": 0.85
    }
    test_result = simulate_granulation(
        mixing_output=sample_mixing_output,
        granulation_time_setpoint=(GRANULATION_TIME_MIN + GRANULATION_TIME_MAX) / 2.0,
        binder_rate_setpoint=(BINDER_RATE_MIN + BINDER_RATE_MAX) / 2.0
    )
    print(test_result)
