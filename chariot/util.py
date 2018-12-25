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
            for i, e in enumerate(data):
                result[i] = func(e)
        return result
    else:
        return func(data)

    return data
