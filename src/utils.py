import numpy as np
import pandas as pd

# =============================== Normalization ===============================
def normalize_midprice(array: np.ndarray, norm_method: str = "Z_Score") -> np.ndarray:
    """
    Normalize the midprice array using Z-Score or Min-Max normalization.

    Args:
        array: np.ndarray:  The array to normalize.
        norm_method: str:   The normalization method to use.

    Returns:
        np.ndarray:         The normalized array.
    """
    if norm_method == "Z_Score":
        return (array-array.mean())/array.std()

    elif norm_method == "MinMax":
        return (array-array.min())/(array.max()-array.min())

# =============================== Normalization Features ===============================
def normalize_quantities(array: np.ndarray, levels: int, norm_method: str = "sqrt") -> np.ndarray:
    """
    Normalize the features array using sqrt or log normalization.

    Args:
        array: np.ndarray:  (2*levels, num_snapshots) The array to normalize.
        levels: int:        The number of levels to normalize (Bids and Asks).
        norm_method: str:   The normalization method to use.

    Returns:
        np.ndarray:         (2*levels, num_snapshots) The normalized array.
    """
    result = np.zeros(array.shape)
    if norm_method == "sqrt":
        result[:levels,:] = -1 * np.sqrt(-1 * array[:levels,:]) - 1
        result[levels:,:] = np.sqrt(array[levels:,:]) + 1
        return result
    elif norm_method == "log":
        result[:levels,:] = -1*np.log(-1*array[:levels,:]+1) - 1
        result[levels:,:] = np.log(array[levels:,:]+1) + 1
        return result
    elif norm_method == 'hypersine':
        pass

# =============================== Descriptive Statistics ===============================
def describe_array(array: np.ndarray) -> pd.DataFrame:
    """
    Describe the numpy array.

    Args:
        array: np.ndarray:  The array to describe.

    Returns:
        pd.DataFrame:       The described array.
    """
    statistics = {
        "count": array.shape[0],
        "mean": round(array.mean(),2),
        "std": round(array.std(),2),
        "min": round(array.min(),2),
        "max": round(array.max(),2),
        '10%': round(np.percentile(array,10),2),
        '25%': round(np.percentile(array,25),2),
        '50%': round(np.median(array),2),
        '75%': round(np.percentile(array,75),2),
        '90%': round(np.percentile(array,90),2)
    }
    
    return pd.DataFrame(statistics, index=[0])

# =============================== Add Midprice ===============================
def add_midprice(centered: np.ndarray, mid_prices: np.ndarray) -> np.ndarray:
    """
    Add the midprice to the centered array.

    Args:
        centered: np.ndarray:   (2*levels, num_snapshots) The centered array.
        mid_prices: np.ndarray: (num_snapshots) The midprice array.

    Returns:
        np.ndarray: (2*levels+1, num_snapshots) The array with the midprice.
    """
    if mid_prices.ndim == 1:
        mid_prices = mid_prices[np.newaxis,:]
    return np.concatenate([mid_prices, centered], axis=0) # (2*levels+1, num_snapshots)


