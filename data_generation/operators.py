import random
import logging

class Operator:
    def __init__(self, name, mixing_bias=1.0, granulation_bias=1.0, drying_bias=1.0, compression_bias=1.0):
        self.name = name
        self.mixing_bias = mixing_bias
        self.granulation_bias = granulation_bias
        self.drying_bias = drying_bias
        self.compression_bias = compression_bias

def get_operator_profiles():
    """
    Returns a list of sample operator profiles.
    """
    operators = [
        Operator("Alice", mixing_bias=1.0, granulation_bias=0.95, drying_bias=1.05, compression_bias=1.0),
        Operator("Bob", mixing_bias=1.05, granulation_bias=1.0, drying_bias=0.95, compression_bias=1.02),
        Operator("Charlie", mixing_bias=0.98, granulation_bias=1.02, drying_bias=1.0, compression_bias=0.97),
    ]
    return operators

def select_operator():
    """
    Randomly selects an operator from the available profiles.
    """
    operators = get_operator_profiles()
    selected = random.choice(operators)
    logging.debug(f"select_operator: Selected operator: {selected.name}")
    return selected
