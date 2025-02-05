import logging
import numpy as np
from data_generation.parameters import *
from data_generation.operators import select_operator

def simulate_mixing(mixing_time_setpoint, mixing_speed_setpoint, previous_output=None):
    """
    Simulates the mixing process given externally injected control parameters.
    
    Parameters:
        mixing_time_setpoint (float): Target mixing time provided by the controller.
        mixing_speed_setpoint (float): Target mixing speed provided by the controller.
        previous_output (dict, optional): Output from the previous process stage (for feedback).

    Returns:
        dict: {
            "status": "success" or "failed",
            "operator": Name of the operator used,
            "mixing_time": Measured mixing time (after noise/drift),
            "mixing_speed": Measured mixing speed (after noise/drift),
            "uniformity_index": Computed quality index,
            "reason": Failure reason (if any),
            "log": Detailed dictionary of intermediate values for feedback.
        }
    """
    log_details = {}

    # Operator selection (could also be injected as a parameter if desired)
    operator = select_operator()
    log_details["selected_operator"] = operator.name

    # Introduce a small machine drift and measurement noise
    machine_drift = np.random.normal(1.0, 0.01)
    log_details["machine_drift"] = machine_drift

    mixing_time_noise = np.random.normal(0, 0.2)
    mixing_speed_noise = np.random.normal(0, 0.5)

    # Compute the actual measurements using the injected setpoints
    mixing_time = mixing_time_setpoint * operator.mixing_bias * machine_drift + mixing_time_noise
    mixing_speed = mixing_speed_setpoint * operator.mixing_bias * machine_drift + mixing_speed_noise

    log_details["injected_mixing_time_setpoint"] = mixing_time_setpoint
    log_details["injected_mixing_speed_setpoint"] = mixing_speed_setpoint
    log_details["measured_mixing_time"] = mixing_time
    log_details["measured_mixing_speed"] = mixing_speed

    # Define ideal values for reference (midpoints of allowed ranges)
    ideal_mixing_time = (MIXING_TIME_MIN + MIXING_TIME_MAX) / 2.0
    ideal_mixing_speed = (MIXING_SPEED_MIN + MIXING_SPEED_MAX) / 2.0
    log_details["ideal_mixing_time"] = ideal_mixing_time
    log_details["ideal_mixing_speed"] = ideal_mixing_speed

    # Compute deviations as a measure of process quality
    time_deviation = abs(mixing_time - ideal_mixing_time) / ideal_mixing_time
    speed_deviation = abs(mixing_speed - ideal_mixing_speed) / ideal_mixing_speed
    log_details["time_deviation"] = time_deviation
    log_details["speed_deviation"] = speed_deviation

    # Compute the uniformity index using a base factor and the deviations
    uniformity_index = UNIFORMITY_INDEX_BASE * (1.0 - (time_deviation + speed_deviation))
    log_details["uniformity_index"] = uniformity_index

    # Determine if the process should be considered a failure based on quality
    if (time_deviation + speed_deviation) > 1.0:
        status = "failed"
        reason = f"Quality too low (combined deviation={(time_deviation + speed_deviation):.2f})"
    else:
        status = "success"
        reason = ""

    result = {
        "status": status,
        "operator": operator.name,
        "mixing_time": mixing_time,
        "mixing_speed": mixing_speed,
        "uniformity_index": uniformity_index,
        "reason": reason,
        "log": log_details
    }

    logging.debug(f"simulate_mixing: Result: {result}")
    return result

if __name__ == "__main__":
    # Example test call using mid-range setpoints:
    test_result = simulate_mixing(
        mixing_time_setpoint=(MIXING_TIME_MIN + MIXING_TIME_MAX) / 2.0,
        mixing_speed_setpoint=(MIXING_SPEED_MIN + MIXING_SPEED_MAX) / 2.0
    )
    print(test_result)
