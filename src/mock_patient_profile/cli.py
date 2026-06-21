"""
CLI for mock_patient_profile
"""

import fire

from mock_patient_profile.main import show_message


class mock_patient_profileCLI:
    def show_message(
        self,
        message: str = "Hello, world!",
    ) -> str:
        """
        CLI interface for show_message.

        Args:
            message (str):
                The message to print.
                Defaults to 'Hello, world!'.

        Returns:
            pd.DataFrame:
                A DataFrame containing the message.
        """

        # prints the message to screen
        print(show_message(message=message))


def trigger() -> None:
    """
    Trigger the CLI to run.
    """
    fire.Fire(mock_patient_profileCLI)
