import numpy as np
from typing import List, Dict, Any, Union

def convert_numpy_value(value: Any) -> Any:
    """Convert a NumPy value to native Python type."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, (np.float32, np.float64)):
        return float(value)
    elif isinstance(value, (np.int32, np.int64)):
        return int(value)
    elif isinstance(value, np.str_):
        return str(value)
    elif isinstance(value, dict):
        return {convert_numpy_value(k): convert_numpy_value(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [convert_numpy_value(v) for v in value]
    return value

def pickable_to_dict(data: List[List[Dict]]) -> Dict[int, Dict]:
    """Convert nested structure with NumPy types to a dictionary with native Python types."""
    result = {}
    
    for env_idx, env_data in enumerate(data):
        result[env_idx] = {}
        for item in env_data:
            for key, value in item.items():
                key = convert_numpy_value(key)
                value = convert_numpy_value(value)
                result[env_idx][key] = value
    
    return result