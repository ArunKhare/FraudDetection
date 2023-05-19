import sys


class FraudDetectionException(Exception):
    def __init__(self, error_message: Exception, error_details: sys):
        super().__init__(error_message)
        self.error_message = FraudDetectionException.get_detailed_error_message(error_message,
                                                                                error_details)

    @staticmethod
    def get_detailed_error_message(error_message: Exception, error_details: sys) -> str:
        """
        error_message:Exception Object
        error_details: Object of sys module
        """
        _, _, exec_tb = error_details.exc_info()
        exception_block_line_number = exec_tb.tb_frame.f_lineno
        try_block_line_number = exec_tb.tb_lineno
        file_name = exec_tb.tb_frame.f_code.co_filename

        error_message = f""" 
        Error occured in script: [{file_name}]at 
        try block line number: [{try_block_line_number}]and
        exception block line number:[{exception_block_line_number}]
        error message [{error_message}]
        """
        return error_message

    def __str__(self) -> str:
        return self.error_message

    def __repr__(self) -> str:
        return FraudDetectionException.__name__.__str__()
