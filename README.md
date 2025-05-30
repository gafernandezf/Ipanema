# Ipanema v1.0 - Simulation Framework
***
### Description
This new version of Ipanema is designed to simplify the implementation process for any needed simulation. It separates the original workflow into a pipeline of simple plugins users can modify. Ipanema handles the remaining steps automatically.

In any specific implementation you will need at least 3 plugins:

1. **Input Plugin:** Data preparation for your model. Any parameter the model might need must be stored in its dictionary.

2. **Model Plugin:** Model definition. FCN declaration, Minuit initialization, etc. Any process needed for a model fitting.

3. **Output Plugin(s):** Handles model execution and processes results for presentation. You may set multiple Output Plugins (e.g., one for printing results, another for plotting data, etc.)

The system contains two main modules inside `src`. `ipanema` (which is the core of the system) and `sdk` (which is a custom Software Development Kit designed for Ipanema).  

#### Ipanema

This module contains 5 main packages:

1. **Config:**  Inside this package resides `config.py`. This file is used to indicate to Ipanema which plugins should be executed. You can also specify any custom file paths used in your simulation. 

2. **Core:** Contains the main pipeline Ipanema uses to dynamically load and execute plugins. You might not need to modify anything in this package.

3. **Input:** Defines the interface which defines the required structure for Input Plugins. It also has a directory named `implementations/`. This directory contains a default and an example implementation of Input Plugins. You may use this directory to store your own Input Plugin implementations.

4. **Model:** Defines the interface which defines the required structure for Model Plugins. It also has a directory named `implementations/`. This directory contains a default and an example implementation of Model Plugins. You may use this directory to store your own Model Plugin implementations.

5. **Output:** Defines the interface which defines the required structure for Output Plugins. It also has a directory named `implementations/`. This directory contains a default implementation of an Output Plugin. You may use this directory to store your own Output Plugin implementations.

#### SDK

This module provides a set of support libraries users may use for their own implementations.

Its present version has 2 main packages: 

- **CUDA Manager:** This package contains different implementations of a `CudaManager` designed for compiling and executing CUDA code in a simple and unified manner. It allows users to use High Performance Computing operations without having knowledge of any particular library. It also supports reduction operations over arrays, as well as element-wise operations. In this version, the implementations of this manager use `PyCuda`.

- **Math Utils:** This package is intended to contain different utilities involving mathematical operations users may need.

***
### The Plugins

This section explains how to properly implement plugins.

Note that for any given plugin a naming convention is used. The file which contains the plugin must have a snake_case name (e.g., `example_name_plugin.py`) while the class implementing the plugin must have the same name in PascalCase (e.g., `ExampleNamePlugin`).

#### InputPlugin

User-defined plugins must implement the `InputPlugin` "interface". The code shown below is a simplification of this plugin.

```python
class InputPlugin():

    @staticmethod
    def get_params() -> dict:
        pass
```

Input Plugins implemented by users may have other methods but must provide their desired parameters in the dictionary returned by `get_params`. 

#### ModelPlugin

User-defined plugins must implement the `ModelPlugin` "interface". The code shown below is a simplification of this plugin.

```python
class ModelPlugin():

    fit_manager: Minuit
    parameters: dict

    def __init__(self, params: dict) -> None:
        self._parameters = params

    def prepare_fit(self) -> None:
        pass
```

Model Plugins implemented by users may add logic inside `__init__` or have other methods, but must use `prepare_fit` to start the model preparation sequence.

#### OutputPlugin

User-defined plugins must implement the `OutputPlugin` "interface". The code shown below is a simplification of this plugin.

```python
class OutputPlugin():

    def generate_results(self, model: ModelPlugin) -> None:
        pass
```

Model Plugins implemented by users may have other methods, but must use `generate_results` to start the sequence of actions leading to the execution of the model fit and results presentation.

***
### How to implement a simulation
...

***
### Example
...

***

### Getting Started
...

***
### Basic Commands

Execution: `hatch run python main.py`

Testing: `hatch run pytest`