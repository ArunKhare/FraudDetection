"""
Fraud Detection Exception Module

This module defines a custom exception class, FraudDetectionException, for handling exceptions in the fraud detection system.

Classes:
    - FraudDetectionException: Custom exception class for fraud detection.

Functions:
    None

Attributes:
    None

Usage:
    Example usage of the FraudDetectionException class:
    
    try:
        # Code that may raise an exception
        raise ValueError("Invalid input data")
    except ValueError as ve:
        # Catch the exception and wrap it with FraudDetectionException
        raise FraudDetectionException(ve, sys) from ve

Classes Description:
    - FraudDetectionException: Custom exception class that extends the base Exception class. It includes additional
      functionality for generating a detailed error message using the sys module.

Methods:
    - __init__(self, error_message: Exception, error_details: sys): Initializes a FraudDetectionException object with
      the original exception object and sys module containing error details.

    - get_detailed_error_message(error_message: Exception, error_details: sys) -> str: Static method that generates a
      detailed error message using information from the original exception and sys module.

    - __str__(self) -> str: String representation of the exception. Returns the detailed error message.

    - __repr__(self) -> str: String representation of the class. Returns the class name.

Example:
    Example usage of the FraudDetectionException class:

    ```python
    try:
        # Code that may raise an exception
        raise ValueError("Invalid input data")
    except ValueError as ve:
        # Catch the exception and wrap it with FraudDetectionException
        raise FraudDetectionException(ve, sys) from ve
    ```
"""

import sys


class FraudDetectionException(Exception):
    def __init__(self, error_message: Exception, error_details: sys):
        """
        Initializes a FraudDetectionException.
        Args:
            error_message (Exception): The original exception object.
            error_details (sys): The sys module object containing error details.
        """
        super().__init__(error_message)
        self.error_message = FraudDetectionException.get_detailed_error_message(
            error_message, error_details
        )

    @staticmethod
    def get_detailed_error_message(error_message: Exception, error_details: sys) -> str:
        """
        Generates a detailed error message.
        Args:
            error_message (Exception): The original exception object.
            error_details (sys): The sys module object containing error details.
        Returns:
            str: Detailed error message.
        """
        _, _, exec_tb = error_details.exc_info()
        exception_block_line_number = exec_tb.tb_frame.f_lineno
        try_block_line_number = exec_tb.tb_lineno
        file_name = exec_tb.tb_frame.f_code.co_filename

        error_message = f""" 
        Error occurred in script: [{file_name}] at 
        try block line number: [{try_block_line_number}] and
        exception block line number: [{exception_block_line_number}]
        error message: [{error_message}]
        """
        return error_message

    def __str__(self) -> str:
        """
        String representation of the exception.
        Returns:
            str: The error message.
        """
        return self.error_message

    def __repr__(self) -> str:
        """
        String representation of the class.
        Returns:
            str: The class name.
        """
        return FraudDetectionException.__name__.__str__()
