from pathlib import Path
from venv import logger
from ipanema.model import ModelPlugin
from iminuit import Minuit
from sdk.cuda_manager.abstract_cuda_manager import CudaManager
from sdk.cuda_manager.implementations.auto_cuda_manager import AutoCudaManager
from sdk.cuda_manager.implementations.interactive_cuda_manager import InteractiveCudaManager
import numpy as np
import math

class SignalPeakModel(ModelPlugin):
    """
    Fit to a signal peak on top of an exponential background.
    
    A signal peak is fitted on top of an exponential background, using an 
    unbinned maximum likelihood fit.
    """

    cuda_manager: CudaManager

    def __init__(self, params):
        """Initializes the model."""
        super().__init__(params)
        self.cuda_manager = InteractiveCudaManager(0, False)

    def prepare_fit(self) -> None:
        """Fits this model using parameters provided during initialization."""

        logger.info("Preparing Fit Manager")

        Ndat = self.parameters["Ndat"]
        self.cuda_manager.add_code_fragment(
            "ipatia",
            Path(
                r"src\ipanema\model\implementations\support_files\ipatia.cu"
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
            Ns = 0.3*Ndat, 
            Nb = 0.7*Ndat, 
            n2 = 1, 
            k = -0.05
        )

        self.fit_manager.limits["mu"] = (5360., 5370.)
        self.fit_manager.limits["sigma"] = (5., 9.)
        self.fit_manager.limits["l"] = (-5., -1.)
        self.fit_manager.limits["beta"] = (-1e-3, 1e-3)
        self.fit_manager.limits["k"] = (-0.05, 0)
        self.fit_manager.limits["Ns"] = (0.1*Ndat, 1.1*Ndat)
        self.fit_manager.limits["Nb"] = (0.1*Ndat, 1.1*Ndat)

        self.fit_manager.fixed["a"] = True
        self.fit_manager.fixed["a2"] = True
        self.fit_manager.fixed["n"] = True
        self.fit_manager.fixed["n2"] = True

        logger.info("Fit Manager Fully Initialized")

    def _generate_fcn(self):

        # Obtaining parameters
        params = self.parameters
        DM = params["DM"]
        Mmax = params["Mmax"]
        Mmin = params["Mmin"]
        mydat = params["mydat"]
        massbins = params["massbins"]
        Ndat = params["Ndat"]
        
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
            integral_ipa = np.sum(ipatia_bins_out[0])*DM

            if k!= 0 : 
                integral_exp = (np.exp(k*Mmax)-np.exp(k*Mmin))*1./k
            else : 
                integral_exp = (Mmax - Mmin)

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
            print(f"\nterm 1 (shape {term1.shape} , type {type(term1)}): {term1}")
            term2 = ipatia_data_out[0] * invint_s * fs
            print(f"ipatia data: {ipatia_data_out[0]}")
            print(f"term2 (shape {term2.shape} , type {type(term2)}): {term2}")
            sum_terms = term1 + term2
            print(f"sum terms (shape {sum_terms.shape} , type {type(sum_terms)}) : {sum_terms}\n\n")
            # Calculate total likelihood
            LL_gpu = self.cuda_manager.single_operation(
                "log", 
                sum_terms
            ) - Nexp
            extendLL =  Ndat*math.log(Nexp) -(Nexp)
            LL = np.float64(
                self.cuda_manager.reduction_operation("sum",LL_gpu)
            ) + extendLL

            chi2 = -2*LL
            return chi2

        return fcn

