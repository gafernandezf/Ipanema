# Ipanema v1.0 - Simulation Framework

***
### Index

- [Ipanema v1.0 - Simulation Framework](#ipanema-v10---simulation-framework)
    - [Index](#index)
    - [Description](#description)
      - [Ipanema](#ipanema)
      - [SDK](#sdk)
    - [The Plugins](#the-plugins)
      - [Naming Convention](#naming-convention)
      - [InputPlugin](#inputplugin)
      - [ModelPlugin](#modelplugin)
      - [OutputPlugin](#outputplugin)
    - [Example](#example)
    - [Getting Started](#getting-started)
      - [1. Clone the repository](#1-clone-the-repository)
      - [2. Install Hatch](#2-install-hatch)
      - [3. Set up the environment](#3-set-up-the-environment)
      - [4. Modify the configuration](#4-modify-the-configuration)
      - [5. Run your simulation](#5-run-your-simulation)
    - [Basic Commands](#basic-commands)

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

Its name stands for **Software Development Kit**. This module provides a set of support libraries users may use for their own implementations.

Its present version has 2 main packages: 

- **CUDA Manager:** This package contains different implementations of a `CudaManager` designed for compiling and executing CUDA code in a simple and unified manner. It allows users to use High Performance Computing operations without having knowledge of any particular library. It also supports reduction operations over arrays, as well as element-wise operations. In this version, the implementations of this manager use `PyCuda`.

- **Math Utils:** This package is intended to contain different utilities involving mathematical operations users may need.

***
### The Plugins

This section explains how to properly implement plugins.

#### Naming Convention

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
### Example

Users who want to implement their own models might follow steps similar to the ones shown below:

1. Implement an Input Plugin. Declare your parameters and return them in a dictionary:

```python
# file: example_input.py
class ExampleInput(InputPlugin):

    ...

    @staticmethod
    def get_params() -> dict:
        params: dict = {}

        param_1 = 1.0
        param_2 = "string"
        ...
        param_n = np.ndarray([1, 2, 3])

        params["param_1"] = param_1
        params["param_2"] = param_2
        ...
        params["param_n"] = param_n
        return params

    ...
```

2. Implement a Model Plugin. Initialize your model using the parameters previously declared:
```python
# file: example_model.py
class ExampleModel(ModelPlugin):

    ...

    def __init__(self, params):
        super().__init__(params)

    def prepare_fit(self) -> None:
        
        self.fit_manager = Minuit(
            self._generate_fcn(), 
            a = 2.4, 
            b = 1.3, 
            c = 3
        )

        self.fit_manager.limits["a"] = (2., 3.)
        self.fit_manager.limits["b"] = (-1., 3.)
        self.fit_manager.limits["c"] = (-5, 15)

    def _generate_fcn(self):
        params = self.parameters
        param_1 = params["param_1"]
        param_n = params["param_n"]

        # Declaring FCN
        def fcn(a, b, c):
            result = a**2 + param_1
            result += b * param_n[1] / np.float(c)

            return result

        return fcn
    
    ...

```

3. Implement an Output Plugin. Execute your model and present the results:
   
```python
# file: example_output.py
class ExampleOutput(OutputPlugin):

    ...

    def generate_results(self, model: ModelPlugin) -> None:
        
        model.fit_manager.migrad()
        model.fit_manager.hesse()

        print(f"\nFit Manager Values: \n{model.fit_manager.values}\n")
        print(f"\nFit Manager Error: \n{model.fit_manager.errors}\n")

    ...

```
4. Modify `config.py` so that Ipanema uses your plugins. Note that Ipanema expects the plugin names without their `.py` file extension in its configuration file:

```python
# file: config.py
CONFIG = {

    "custom_paths": ['if\\your\\plugins\\outside\\implementations\\directories'],

    "input": "example_input",

    "model": "example_model",

    "outputs": [
        "example_output"
    ],
}
```  

5. Access the root directory of this project and run your simulation with the **Execution** command provided in **Basic Commands** section.

***
### Getting Started

#### 1. Clone the repository
```bash
git clone https://github.com/GabrielFernandez1/Ipanema.git
```

#### 2. Install Hatch
If your do not have hatch installed in your computer:
```bash
pip install hatch
```

#### 3. Set up the environment
Use hatch to create and activate a development environment:
```bash
hatch shell
```

#### 4. Modify the configuration
Adjust `config.py` to your necessities.

#### 5. Run your simulation
Use the **Execution** command in **Basic Commands** section.

***
### Basic Commands

Execution: `hatch run python src/main.py`

Testing: `hatch run pytest`