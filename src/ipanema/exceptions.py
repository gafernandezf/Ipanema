class IpanemaException(Exception):
    """Base exception for Ipanema specialized errors."""

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