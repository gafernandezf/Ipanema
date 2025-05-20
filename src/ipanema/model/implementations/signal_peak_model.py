from ipanema.model import ModelPlugin
from iminuit import Minuit
from sdk.cuda_manager import CudaManager
from sdk.cuda_manager import AutoCudaManager
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
        self.cuda_manager = AutoCudaManager()

    def prepare_fit(self) -> None:
        """Fits this model using parameters provided during initialization."""
        Ndat = self.parameters["Ndat"]
        self.cuda_manager.add_code_fragment("Path of Ipatia.cu")

        # Minuit Fit Manager Initialization
        self._fit_manager = Minuit(
            self._generate_fcn(), 
            mu = 5365., 
            limit_mu = (5360., 5370.), 
            sigma = 7., 
            limit_sigma= (5.,9.), 
            l = -3., 
            limit_l= (-5.,-1.),
            beta = 0., 
            limit_beta = (-1e-03,1e-03), 
            a = 3., 
            fix_a = True, 
            n = 1,
            a2 = 6, 
            limit_k = (-0.05,0), 
            limit_Ns = (0.1*Ndat, 1.1*Ndat), 
            limit_Nb = (0.1*Ndat,1.1*Ndat),
            Ns = 0.3*Ndat, 
            Nb = 0.7*Ndat, 
            fix_a2 = True, 
            n2 = 1, 
            fix_n = True, 
            fix_n2 = True
        )

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
            ipatia_bins_out: list = self.cuda_manager.run_program(
                "ipatia",
                [1],
                {1: [[len(massbins)], np.double]},
                (1000,1,1),
                (len(mydat)/1000,1),
                [
                    massbins, 
                    np.empty_like(massbins), 
                    mu, 
                    sigma, 
                    l, 
                    beta, 
                    a, 
                    n, 
                    a2, 
                    n2
                ]
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
                "ipatia",
                [1],
                {1: [[len(massbins)], np.double]},
                (512,1,1),
                (len(mydat)/512,1),
                [
                    mydat, 
                    np.empty_like(mydat), 
                    mu, 
                    sigma, 
                    l, 
                    beta, 
                    a, 
                    n, 
                    a2, 
                    n2
                ]
            )
            # Exponential background
            bkg_gpu = self.cuda_manager.single_operation("exp",[k*mydat])

            # Calculate total likelihood
            LL_gpu = self.cuda_manager.single_operation(
                "log", 
                [fs*invint_s*ipatia_data_out[0] + fb*invint_b*bkg_gpu]
            ) - Nexp
            extendLL =  Ndat*math.log(Nexp) -(Nexp)
            LL = np.float64(
                self.cuda_manager.reduction_operation("sum",LL_gpu)
            ) + extendLL

            chi2 = -2*LL
            return chi2

        return fcn

