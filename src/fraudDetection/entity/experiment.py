""" 
This module contains the Experiment tracking class.

Class:
    Experiment
        A class to initiate new experiments and track their status.

Attributes:
    running_status (bool): Indicates the status of experiments ('running' or 'complete').

Methods:
    __new__:
        Checks the status of the experiment before creating a new instance.

    __init__:
        Initializes the experiment with an experiment ID and status.

Usage:
    To use this module, create an instance of the Experiment class with a unique experiment ID.
"""


class Experiment:
    """A class to initiate new experiments and track their status."""

    running_status = False

    def __new__(cls, *args, **kwargs):
        """check the status of experiment
        Args:
            cls
                the class for which the object is created
            *args
                Variable length argument list.
            **kwargs
                Arbitrary keyword arguments.
        Returns:
            An instance of the Experiment class created by the superclass.
        Raises:
            Exception: if Experiment is already running
        """
        if Experiment.running_status:
            raise RuntimeError(
                "Experiment is alrady running hence new experiment can not be created"
            )
        return super(Experiment, cls).__new__(cls, *args, **kwargs)

    def __init__(self, experiment_id) -> None:
        """Initialize the experiment with an experiment ID and status.
        Args:
            experiment_id: Unique identifier for the experiment.
        """
        self.experiment_id = experiment_id
        self.running_status = Experiment.running_status
