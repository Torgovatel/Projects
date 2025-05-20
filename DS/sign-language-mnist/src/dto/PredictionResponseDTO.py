from pydantic import BaseModel, constr, condecimal
from typing import Dict


class PredictionResponseDTO(BaseModel):
    """
    A data transfer object (DTO) representing a prediction response.

    This model contains the predicted label and a dictionary of probabilities
    for each class.

    :param prediction: The predicted label, a single uppercase letter (A-Z).
    :type prediction: str
    :param probabilities: A dictionary of class probabilities where keys are
                          uppercase letters (A-Z) and values are decimal numbers
                          between 0 and 1, inclusive.
    :type probabilities: Dict[str, float]

    :returns: A structured response containing prediction and probabilities.
    :rtype: PredictionResponseDTO
    """

    prediction: constr(min_length=1, max_length=1, pattern=r"^[A-Z]$")
    probabilities: Dict[
        constr(min_length=1, max_length=1, pattern=r"^[A-Z]$"), condecimal(ge=0, le=1)
    ]

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "prediction": "B",
                "probabilities": {
                    "A": 0.1,
                    "B": 0.85,
                    "C": 0.05,
                    "D": 0.0,
                    "E": 0.0,
                    "F": 0.0,
                    "G": 0.0,
                    "H": 0.0,
                    "I": 0.0,
                    "J": 0.0,
                    "K": 0.0,
                    "L": 0.0,
                    "M": 0.0,
                    "N": 0.0,
                    "O": 0.0,
                    "P": 0.0,
                    "Q": 0.0,
                    "R": 0.0,
                    "S": 0.0,
                    "T": 0.0,
                    "U": 0.0,
                    "V": 0.0,
                    "W": 0.0,
                    "X": 0.0,
                    "Y": 0.0,
                    "Z": 0.0,
                },
            }
        }
