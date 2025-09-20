
import pandas as pd 
import numpy as np 

def check_config(config: pd.DataFrame):
    """
    Validates the format and content of a dataset configuration DataFrame.
    
    Parameters
    ----------
    config : pd.DataFrame
        Configuration DataFrame with columns: image_path, class_id
        and attrs containing metadata about the dataset.
        
    Raises
    ------
    ValueError
        If any validation check fails.
    """
    
    # Check required attrs
    if "dataset_folder" not in config.attrs or config.attrs["dataset_folder"] is None:
        raise ValueError("Dataset folder is not set in DataFrame attrs")
    
    # Check required columns exist
    required_columns = ["image_path", "class_id"]
    missing_columns = [col for col in required_columns if col not in config.columns]
    if missing_columns:
        raise ValueError(f"Required columns missing: {missing_columns}")
    
    # Check DataFrame is not empty
    if len(config) == 0:
        raise ValueError("Configuration DataFrame is empty")
    
    if not pd.api.types.is_numeric_dtype(config["class_id"]):
        raise ValueError("Column 'class_id' must contain numeric values")
    
    # Check image_path is string-like (object dtype in pandas often contains strings)
    if not (pd.api.types.is_string_dtype(config["image_path"]) or 
            pd.api.types.is_object_dtype(config["image_path"])):
        raise ValueError("Column 'image_path' must contain string values")
    
    # Check for null values in critical columns
    for col in required_columns:
        if config[col].isnull().any():
            raise ValueError(f"Column '{col}' contains null values")
    
    # Check for unique indices (each image has unique ID)
    if not config.index.is_unique:
        raise ValueError("DataFrame index must be unique (each image needs a unique ID)")
    
    # Additional sanity checks
    if config["class_id"].min() < 0:
        raise ValueError("class_id should not contain negative values")
    
    # Check that class_id are integers (even if stored as float)
    if not np.all(config["class_id"] == config["class_id"].astype(int)):
        raise ValueError("class_id should be integer values")
    
    # Check image paths are not empty strings
    if (config["image_path"] == "").any():
        raise ValueError("image_path column contains empty strings")
    

    print
if __name__ == "__main__": 
    dsconfig = pd.read_parquet("datasets/train/configs/nanoplace_basic/config.parquet")
    check_config(dsconfig)