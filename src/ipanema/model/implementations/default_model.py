from ipanema.model import ModelPlugin
from iminuit import Minuit


class DefaultModel(ModelPlugin):
    """
    Concrete implementation of a ModelPlugin using a simple example function.

    This model fits the function (x - 3)**2 using Minuit.
    """

    def __init__(self, params):
        """
        Initialize the DefaultModel with the given parameters.

        Arguments:
            params (dict): Parameters to be used during model initialization.
        """
        super().__init__(params)

    def prepare_fit(self) -> None:
        """
        Prepare the fit manager using the provided parameters.

        Initializes the Minuit fit manager with the generated FCN 
        (function to minimize).
        """
        self.fit_manager = Minuit(self._generate_fcn(), x=1)

    def _generate_fcn(self):
        """
        Generate the function to be minimized during fitting.

        Returns:
            Callable: The function (x - 3)**2.
        """
        def fcn(x):
            return (x - 3)**2

        return fcn

