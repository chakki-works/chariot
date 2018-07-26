import pandas as pd
import numpy as np


def apply_map(data, func, inplace=False):

    if isinstance(data, pd.DataFrame):
        result = data.applymap(func)
        return result
    elif isinstance(data, pd.Series):
        result = data.apply(func)
        return result
    elif isinstance(data, dict):
        result = {} if inplace else data
        for k in data:
            result[k] = [func(row) for row in data[k]]
        return result
    elif isinstance(data, (list, tuple)):
        result = ([0] * len(data)) if inplace else data
        for i in range(len(data)):
            row = data[i]
            if len(row) > 0 and isinstance(row[0], (list, tuple)):
                # row has multiple columns
                _row = []
                for column in row:
                    _row.append(func(column))
                result[i] = _row
            else:
                result[i] = func(row)

    return data
