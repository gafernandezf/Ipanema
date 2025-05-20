from ipanema.input.input_plugin import InputPlugin
import pickle
import numpy as np

class UserInput(InputPlugin):

    @staticmethod
    def getParams() -> dict:
        """Prepares data for a model initialization."""

        sd = "float64"
        dtype = getattr(np, sd)

        params: dict = {}

        data = pickle.load("path data_SnB.ext")
        mydat = dtype(data[0])
        Ndat = len(mydat)
        massbins = dtype(data[1])
        DM = dtype(massbins[1] - massbins[0])
        Mmax = max(massbins)
        Mmin = min(massbins)

        params[mydat] = mydat
        params[Ndat] = Ndat
        params[DM] = DM
        params[Mmax] = Mmax
        params[Mmin] = Mmin
        params[massbins] = massbins
        
        
        return params