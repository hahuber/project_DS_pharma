import logging
from data_generation import mixing, granulation, drying, compression
from data_generation.parameters import *
from data_generation.utils import setup_logging, setup_random_seed

def run_simulation():
    """
    Runs the full simulation of the tablet manufacturing process using externally injected control parameters.
    Each stage receives setpoints, and detailed logs are provided for feedback.
    """
    logging.info("Starting simulation of tablet manufacturing process.")
    results = {}

    # --- Mixing Stage ---
    mixing_time_setpoint = (MIXING_TIME_MIN + MIXING_TIME_MAX) / 2.0
    mixing_speed_setpoint = (MIXING_SPEED_MIN + MIXING_SPEED_MAX) / 2.0
    mixing_output = mixing.simulate_mixing(mixing_time_setpoint, mixing_speed_setpoint)
    results["mixing"] = mixing_output
    if mixing_output.get("status") != "success":
        results["final_status"] = "failed"
        logging.error(f"Mixing stage failed: {mixing_output.get('reason')}")
        return results

    # --- Granulation Stage ---
    granulation_time_setpoint = (GRANULATION_TIME_MIN + GRANULATION_TIME_MAX) / 2.0
    binder_rate_setpoint = (BINDER_RATE_MIN + BINDER_RATE_MAX) / 2.0
    granulation_output = granulation.simulate_granulation(mixing_output, granulation_time_setpoint, binder_rate_setpoint)
    results["granulation"] = granulation_output
    if granulation_output.get("status") != "success":
        results["final_status"] = "failed"
        logging.error(f"Granulation stage failed: {granulation_output.get('reason')}")
        return results

    # --- Drying Stage ---
    drying_temp_setpoint = (DRYING_TEMP_MIN + DRYING_TEMP_MAX) / 2.0
    drying_output = drying.simulate_drying(granulation_output, drying_temp_setpoint)
    results["drying"] = drying_output
    if drying_output.get("status") != "success":
        results["final_status"] = "failed"
        logging.error(f"Drying stage failed: {drying_output.get('reason')}")
        return results

    # --- Compression Stage ---
    compression_pressure_setpoint = (COMPRESSION_PRESSURE_MIN + COMPRESSION_PRESSURE_MAX) / 2.0
    tablet_hardness_setpoint = (TABLET_HARDNESS_MIN + TABLET_HARDNESS_MAX) / 2.0
    compression_output = compression.simulate_compression(drying_output, compression_pressure_setpoint, tablet_hardness_setpoint)
    results["compression"] = compression_output
    if compression_output.get("status") != "success":
        results["final_status"] = "failed"
        logging.error(f"Compression stage failed: {compression_output.get('reason')}")
        return results

    results["final_status"] = "success"
    logging.info("Simulation completed successfully.")
    return results

if __name__ == "__main__":
    setup_logging()
    setup_random_seed(None)
    simulation_results = run_simulation()
    print(simulation_results)
