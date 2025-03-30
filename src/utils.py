import os
import sys
import dill
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging

def  save_object(file_path, obj):
    """
    Save the object to a file using dill.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
            logging.info(f"Object saved at {file_path}")
    except Exception as e:
        raise CustomException(e, sys) from e