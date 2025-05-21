from ipanema.input.input_plugin import InputPlugin

class DefaultInput(InputPlugin):

    @staticmethod
    def get_params() -> dict:
        """Prepares data for a model initialization."""
        params: dict = {}
        return params