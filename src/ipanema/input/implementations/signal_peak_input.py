from pathlib import Path
from venv import logger
from ipanema.input.input_plugin import InputPlugin
import pickle
import numpy as np

class SignalPeakInput(InputPlugin):

    @staticmethod
    def get_params() -> dict:
        """Prepares data for a model initialization."""

        logger.info("Processing Parameters")

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
        Ndat = len(mydat)
        massbins = dtype(data[1])
        DM = dtype(massbins[1] - massbins[0])
        Mmax = max(massbins)
        Mmin = min(massbins)

        params["mydat"] = mydat
        params["Ndat"] = Ndat
        params["DM"] = DM
        params["Mmax"] = Mmax
        params["Mmin"] = Mmin
        params["massbins"] = massbins

        logger.info("Parameters Fully Processed")
        
        return params