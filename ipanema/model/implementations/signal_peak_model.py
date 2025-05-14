from ipanema.model import ModelPlugin

class SignalPeakModel(ModelPlugin):
    """
    Fit to a signal peak on top of an exponential background.
    
    A signal peak is fitted on top of an exponential background, using an 
    unbinned maximum likelihood fit.
    
    """

    def __init__(self, params: dict) -> None:
        """Initializes the model."""
        pass

    def prepare_fit() -> None:
        """Fits this model using parameters provided during initialization."""
        pass