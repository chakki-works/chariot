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
        result = data if inplace else {}
        for k in list(data.keys()):
            result[k] = [func(row) for row in data[k]]
        return result
    elif isinstance(data, (list, tuple)):
        result = data if inplace else ([0] * len(data))

        # 2d array
        if len(data) > 0 and isinstance(data[0], (list, tuple)):
            data_dim = len(np.array(data[0]).shape)
            for i, row in enumerate(data):
                if data_dim == 1:
                    result[i] = func(row)
                else:
                    # row has multiple column
                    _row = []
                    for column in row:
                        _row.append(func(column))
                    result[i] = _row
        else:
            result = func(data)
        return result
    else:
        return func(data)

    return data
