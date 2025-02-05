# tests/test_data_collector.py
import os
from digital_twin.data_collector import collect_data

def test_csv_creation(tmp_path):
    # Use a temporary file for testing
    # output_file = tmp_path / "test_simulation_results.csv"
    output_file = "simulation_results_test.csv"  # Saves to your project root
    collect_data(num_simulations=10, output_file=str(output_file))
    
    # Check that the file exists
    assert os.path.exists(output_file)
    
    # Read the actual header from the file
    with open(output_file, 'r') as f:
        header = f.readline().strip().split(",")

    # Define the expected column names as a set (order doesn't matter)
    expected_columns = {
        "mixing_status", "mixing_operator", "mixing_time", "mixing_speed", "uniformity_index",
        "granulation_status", "granulation_operator", "granulation_time", "binder_rate", "granule_density",
        "drying_status", "drying_operator", "drying_temp", "moisture_content",
        "compression_status", "compression_operator", "comp_pressure", "tablet_hardness",
        "weight_variation", "dissolution", "yield_percent",
        "final_status", "failure_reason"
    }

    # Convert the actual header to a set and compare
    assert set(header) == expected_columns, f"Header mismatch!\nExpected: {expected_columns}\nGot: {set(header)}"
