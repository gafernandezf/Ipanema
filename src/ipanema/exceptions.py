class IpanemaException(Exception):
    """Base exception for Ipanema specialized errors."""

    def __init__(self, message: str, exception: Exception = None):
        super().__init__(f"{message}: {exception}")
        self.exception = exception

class IpanemaImportError(IpanemaException):
    """Errors importing modules or classes."""
    pass

class IpanemaInitializationError(IpanemaException):
    """Errors during input preparation."""
    pass

class IpanemaFittingError(IpanemaException):
    """Errors furing fit manager definition."""
    pass

class IpanemaOutputError(IpanemaException):
    """Errors involving the model execution or results presentation."""
    pass