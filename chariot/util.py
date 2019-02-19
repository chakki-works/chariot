import pandas as pd
import numpy as np


def apply_map(data, func, inplace=False):

    if isinstance(data, pd.DataFrame):
        # apply and apply_map call function twice.
        result = data if inplace else {}
        for c in data.columns:
            s = list(map(func, data[c].values))
            if inplace:
                result.loc[:, c] = s
            else:
                result[c] = s
        if not inplace:
            result = pd.DataFrame.from_dict(result)
        return result
    elif isinstance(data, pd.Series):
        if inplace:
            for i, e in enumerate(map(func, data.values)):
                data.iat[i] = e
            return data
        else:
            result = pd.Series(list(map(func, data.values)))
            return result
    elif isinstance(data, np.ndarray):
        def _apply_map(x):
            return apply_map(x, func, inplace=inplace)

        if len(data.shape) > 1:
            result = np.apply_along_axis(_apply_map, -1, data)
            return result
        else:
            _data = pd.Series(data)
            return apply_map(_data, func, inplace=inplace)

    elif isinstance(data, dict):
        result = data if inplace else {}
        for k in list(data.keys()):
            result[k] = [func(row) for row in data[k]]
        return result
    elif isinstance(data, (list, tuple)):
        result = data if inplace else ([0] * len(data))
        for i, row in enumerate(data):
            if isinstance(row, (tuple, list)) and len(row) > 0\
                 and isinstance(row[0], (tuple, list)):
                # multi-column list
                result[i] = [func(c) for c in row]
            else:
                result[i] = func(row)
        return result
    else:
        return func(data)

    return data
