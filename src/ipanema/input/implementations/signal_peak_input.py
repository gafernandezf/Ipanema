from pathlib import Path
from ipanema.input.input_plugin import InputPlugin
import pickle
import numpy as np

class SignalPeakInput(InputPlugin):
    """
    Plugin dedicated to the parameter processing and parsing for a fit to a 
    signal peak on top of an exponential background.  
    """

    @staticmethod
    def get_params() -> dict:
        """
        Prepares data for a model initialization.
        
        Defines a dictionary containing the parameters needed by 
        SignalPeakModel.

        Returns:
            dict: Dictionary formed by the expected parameters.
        """

        sd = "float64"
        dtype = getattr(np, sd)

        params: dict = {}

        with open(
            Path(
                r"src\ipanema\input\implementations\support_files\data_SnB.ext"
            ), 
            "rb"
        ) as file:
            data = pickle.load(file, encoding="latin1")
        mydat = dtype(data[0])
        n_dat = len(mydat)
        massbins = dtype(data[1])
        d_m = dtype(massbins[1] - massbins[0])
        m_max = max(massbins)
        m_min = min(massbins)

        params["mydat"] = mydat
        params["n_dat"] = n_dat
        params["d_m"] = d_m
        params["m_max"] = m_max
        params["m_min"] = m_min
        params["massbins"] = massbins
        
        return params