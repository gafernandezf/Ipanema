from ipanema.model import ModelPlugin
from iminuit import Minuit


class DefaultModel(ModelPlugin):
    """Fit of an example function."""


    def __init__(self, params):
        """Initializes the model."""
        super().__init__(params)

    def prepare_fit(self) -> None:
        """Fits this model using parameters provided during initialization."""
        
        # Minuit Fit Manager Initialization
        self.fit_manager = Minuit(self._generate_fcn(), 1)

    def _generate_fcn(self):
        
        # Declaring FCN
        def fcn(x):
            return x

        return fcn

