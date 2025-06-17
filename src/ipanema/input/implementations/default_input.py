from ipanema.input.input_plugin import InputPlugin

class DefaultInput(InputPlugin):

    @staticmethod
    def get_params() -> dict:
        """
        Prepare and return the default parameters for model initialization.

        Returns:
            dict: An empty dictionary representing the default parameters.
        """
        params: dict = {}
        return params