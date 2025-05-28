from pathlib import Path
from ipanema.model import ModelPlugin
from iminuit import Minuit
from sdk.cuda_manager.abstract_cuda_manager import CudaManager
from sdk.cuda_manager.implementations.auto_cuda_manager import AutoCudaManager
from sdk.cuda_manager.implementations.interactive_cuda_manager import (
    InteractiveCudaManager
)
import numpy as np
import math

class SignalPeakModel(ModelPlugin):
    """
    Plugin dedicated to the preparation of a fit to a signal peak on top of 
    an exponential background.
    
    A signal peak is fitted on top of an exponential background, using an 
    unbinned maximum likelihood fit.

    Atributtes:
        fit_manager (Minuit): Function minimizer and error computer used during
            the fitting process
        parameters (dict): Dictionary containing the parameters required during 
            'fit_manager' initialization. 
        cuda_manager (CudaManager): CUDA handler used for the HPC calculus
            during FCN execution.
    """

    _cuda_manager: CudaManager

    def __init__(self, params):
        """Initializes the model."""
        super().__init__(params)
        self.cuda_manager = InteractiveCudaManager(None, False)

    def prepare_fit(self) -> None:
        """
        Prepares 'fit_manager' for its use.
        
        Initializes 'fit_manager' using the 'parameters' previously provided.
        """
        n_dat = self.parameters["n_dat"]
        self.cuda_manager.add_code_fragment(
            "ipatia",
            Path(
                r"src\ipanema\model\implementations\_support_files\ipatia.cu"
            )
        )

        # Minuit Fit Manager Initialization
        self.fit_manager = Minuit(
            self._generate_fcn(), 
            mu = 5365., 
            sigma = 7., 
            l = -3., 
            beta = 0., 
            a = 3., 
            n = 1,
            a2 = 6, 
            Ns = 0.3*n_dat, 
            Nb = 0.7*n_dat, 
            n2 = 1, 
            k = -0.05
        )

        self.fit_manager.limits["mu"] = (5360., 5370.)
        self.fit_manager.limits["sigma"] = (5., 9.)
        self.fit_manager.limits["l"] = (-5., -1.)
        self.fit_manager.limits["beta"] = (-1e-3, 1e-3)
        self.fit_manager.limits["k"] = (-0.05, 0)
        self.fit_manager.limits["Ns"] = (0.1*n_dat, 1.1*n_dat)
        self.fit_manager.limits["Nb"] = (0.1*n_dat, 1.1*n_dat)

        self.fit_manager.fixed["a"] = True
        self.fit_manager.fixed["a2"] = True
        self.fit_manager.fixed["n"] = True
        self.fit_manager.fixed["n2"] = True

    def _generate_fcn(self):
        """Method responsible for the definition of the FCN."""

        # Obtaining parameters
        params = self.parameters
        d_m = params["d_m"]
        m_max = params["m_max"]
        m_min = params["m_min"]
        mydat = params["mydat"]
        massbins = params["massbins"]
        n_dat = params["n_dat"]
        
        # Declaring FCN
        def fcn(mu, sigma, l, beta, a, n, a2, n2, k, Ns, Nb):
            # Calling ipatia for mass_bins
            grid_x = math.ceil(len(mydat) / 512)
            grid = (grid_x, 1)
            block = (512, 1, 1)
            ipatia_bins_out: list = self.cuda_manager.run_program(
                "Ipatia",
                [1],
                {1: [(len(massbins),), np.double]},
                block,
                grid,
                massbins, 
                np.empty_like(massbins), 
                mu, 
                sigma, 
                l, 
                beta, 
                a, 
                n, 
                a2, 
                n2,
                len(mydat)
                
            )
            integral_ipa = np.sum(ipatia_bins_out[0])*d_m

            if k!= 0 : 
                integral_exp = (np.exp(k*m_max)-np.exp(k*m_min))*1./k
            else : 
                integral_exp = (m_max - m_min)

            invint_b = 1./integral_exp
            invint_s = 1./integral_ipa
            Nexp = Ns+Nb
            fs = np.float64(Ns*1./Nexp)
            fb = np.float64(1.-fs)

            # Calling ipatia for my_dat
            ipatia_data_out: list = self.cuda_manager.run_program(
                "Ipatia",
                [1],
                {1: [(len(massbins)), np.double]},
                block,
                grid,
                mydat, 
                np.empty_like(mydat), 
                mu, 
                sigma, 
                l, 
                beta, 
                a, 
                n, 
                a2, 
                n2,
                len(mydat)
            )
            # Exponential background
            bkg_gpu = self.cuda_manager.single_operation("exp",k*mydat)
            term1 = bkg_gpu * invint_b * fb
            term2 = ipatia_data_out[0] * invint_s * fs
            sum_terms = term1 + term2
            # Calculate total likelihood
            LL_gpu = self.cuda_manager.single_operation(
                "log", 
                sum_terms
            ) - Nexp
            extendLL =  n_dat*math.log(Nexp) -(Nexp)
            LL = np.float64(
                self.cuda_manager.reduction_operation("sum",LL_gpu)
            ) + extendLL

            chi2 = -2*LL
            return chi2

        return fcn

    @property
    def cuda_manager(self) -> dict:
        """Getter for cuda_manager property."""
        return self._cuda_manager
    
    @cuda_manager.setter
    def cuda_manager(self, manager: CudaManager):
        """Setter for cuda_manager property."""
        self._cuda_manager = manager