import logging
import numpy as np
from data_generation.parameters import *

def simulate_drying(granulation_output, drying_temp_setpoint):
    """
    Simulates the drying process using externally injected control parameters.
    
    Parameters:
        granulation_output (dict): Output from the granulation stage containing (at minimum)
                                   "status", "operator", and "granule_density".
        drying_temp_setpoint (float): The target drying temperature provided by the controller.
    
    Returns:
        dict: {
            "status": "success" or "failed",
            "operator": Operator name (propagated from granulation_output),
            "drying_temp": Measured drying temperature (after drift and environmental adjustments),
            "moisture_content": Computed moisture content,
            "reason": Failure reason if any,
            "log": Detailed dictionary of intermediate values for feedback.
        }
    """
    log_details = {}

    if granulation_output.get("status") != "success":
        return granulation_output

    operator = granulation_output["operator"]
    log_details["operator"] = operator
    log_details["granule_density"] = granulation_output.get("granule_density")

    # Sample environmental conditions (simulate slight fluctuations around the mean)
    ambient_temp = np.random.normal(AMBIENT_TEMP_MEAN, AMBIENT_TEMP_STD)
    humidity = np.random.normal(HUMIDITY_MEAN, HUMIDITY_STD)
    log_details["ambient_temp"] = ambient_temp
    log_details["humidity"] = humidity

    # Introduce machine drift for drying
    machine_drift = np.random.normal(1.0, 0.01)
    log_details["machine_drift"] = machine_drift

    # Noise for temperature measurement
    drying_temp_noise = np.random.normal(0, 0.5)

    # Compute the measured drying temperature using the injected setpoint, drift, and environmental adjustments.
    ambient_temp_adjustment = (ambient_temp - AMBIENT_TEMP_MEAN) * 0.1
    humidity_adjustment = (humidity - HUMIDITY_MEAN) * 0.05
    measured_drying_temp = drying_temp_setpoint * machine_drift + ambient_temp_adjustment + humidity_adjustment + drying_temp_noise

    log_details["injected_drying_temp_setpoint"] = drying_temp_setpoint
    log_details["measured_drying_temp"] = measured_drying_temp

    # Define ideal values for moisture content and drying temperature
    ideal_moisture = (MOISTURE_CONTENT_MIN + MOISTURE_CONTENT_MAX) / 2.0
    ideal_drying_temp = (DRYING_TEMP_MIN + DRYING_TEMP_MAX) / 2.0
    log_details["ideal_drying_temp"] = ideal_drying_temp
    log_details["ideal_moisture"] = ideal_moisture

    # Compute moisture content as a function of drying efficiency.
    moisture_noise = np.random.normal(0, 0.2)
    moisture_content = ideal_moisture * (ideal_drying_temp / measured_drying_temp) + moisture_noise
    log_details["moisture_content"] = moisture_content

    # Check for failure condition: if moisture_content is excessively high.
    if moisture_content > MOISTURE_CONTENT_FAILURE:
        result = {
            "status": "failed",
            "reason": f"High moisture content ({moisture_content:.2f} > {MOISTURE_CONTENT_FAILURE})",
            "operator": operator,
            "log": log_details
        }
        logging.debug(f"simulate_drying: Failing with result: {result}")
        return result

    result = {
        "status": "success",
        "operator": operator,
        "drying_temp": measured_drying_temp,
        "moisture_content": moisture_content,
        "reason": "",
        "log": log_details
    }
    logging.debug(f"simulate_drying: Success with result: {result}")
    return result

if __name__ == "__main__":
    # Example test call:
    sample_granulation_output = {
        "status": "success",
        "operator": "Alice",
        "granule_density": 1.5
    }
    test_result = simulate_drying(
        granulation_output=sample_granulation_output,
        drying_temp_setpoint=(DRYING_TEMP_MIN + DRYING_TEMP_MAX) / 2.0
    )
    print(test_result)
