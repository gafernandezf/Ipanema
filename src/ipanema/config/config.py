# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
# Configuration dictionary used to initialize and run the plugin-based system.
#
# Fields:
# - custom_paths (list[str]): Optional paths to be added for dynamic module 
#       loading.
# - input (str): Name of the specific InputPlugin module to load.
# - model (str): Name of the specific ModelPlugin module to load.
# - outputs (list[str]): List of OutputPlugin modules to be used.
# -----------------------------------------------------------------------------
CONFIG = {

    "custom_paths": ['C:\\Users\\gabri\\OneDrive\\Escritorio\\MUEI\\3C'],

    "input": "signal_peak_input",

    "model": "signal_peak_model",

    "outputs": [
        "command_line_output"
    ],
}