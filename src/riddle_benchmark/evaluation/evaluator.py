import unicodedata
from riddle_benchmark.dataset.schema import Riddle

class Evaluator:
    """
    Evaluator for riddle answers.
    """

    @staticmethod
    def normalize(text: str) -> str:
        """
        Normalize the text for comparison.

        Steps:
        1. NFKC normalization (unifies fullwidth/halfwidth, etc.)
        2. Lowercase conversion
        3. Remove whitespace and newlines

        Note: This normalization preserves basic symbols like '□' or '◯' because:
        - NFKC converts compatible chars but keeps standard symbols.
        - 'text.split()' removes whitespace but not symbols.

        Args:
            text: The input text to normalize.

        Returns:
            The normalized string.
        """
        # NFKC normalization
        text = unicodedata.normalize("NFKC", text)

        # Lowercase
        text = text.lower()

        # Remove whitespace
        text = "".join(text.split())

        return text

    @classmethod
    def evaluate(cls, prediction: str, riddle: Riddle) -> bool:
        """
        Evaluate if the prediction matches any of the acceptable answers.

        Args:
            prediction: The model's predicted answer.
            riddle: The riddle object containing acceptable answers.

        Returns:
            True if the prediction is correct, False otherwise.
        """
        normalized_prediction = cls.normalize(prediction)

        for answer in riddle.acceptable_answers:
            if normalized_prediction == cls.normalize(answer):
                return True

        return False
