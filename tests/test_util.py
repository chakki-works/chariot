import unittest
import copy
import numpy as np
import pandas as pd
from chariot.util import apply_map


class TestUtil(unittest.TestCase):

    def test_apply_map(self):
        self._test_apply_map(False)

    def test_apply_map_inplace(self):
        self._test_apply_map(True)

    def _test_apply_map(self, inplace):
        value_dict = {
            "column1": [1, 2, 3, 4, 5],
            "column2": [4, 5, 6, 7, 8],
        }
        array_dict = {
            "column1": [[0, 1], [2, 3, 4], [4, 5]],
            "column2": [[3, 4], [5, 6], [2, 1, 9]],
        }

        df = pd.DataFrame(value_dict)
        array_df = pd.DataFrame(array_dict)

        series = pd.Series(value_dict["column1"])
        array_series = pd.Series(array_dict["column1"])

        value_list = [1, 2, 3, 4, 5]
        array_list = [[0, 1], [2, 3, 2], [4, 5, 1]]
        multi_array_list = [[[3, 1], [2, 3, 1]], [[4, 5], [4, 5]]]

        test_dict = {
            "value": [df, series, value_dict, value_list, None],
            "array": [array_df, array_series, array_dict,
                      array_list, multi_array_list]
        }
        kinds = ["DataFrame", "Series", "dict", "list", "multi-array"]

        for k in test_dict:
            if k == "value":
                func = lambda x: x * 2
            else:
                func = lambda x: sum(x)

            for _k, d in zip(kinds, test_dict[k]):
                if d is None:
                    continue
                print("{}-{}".format(k, _k))
                _d = copy.deepcopy(d)
                result = apply_map(_d, func, inplace)
                if inplace:
                    result = _d
                if _k == "DataFrame":
                    print(result)
                    self.assertEqual(tuple(self.flatten(result)),
                                     tuple(map(func, self.flatten(d))))
                elif _k == "Series":
                    self.assertEqual(tuple(self.flatten(result)),
                                     tuple(map(func, self.flatten(d))))
                elif _k == "dict":
                    for kx in result:
                        self.assertEqual(tuple(self.flatten(result[kx])),
                                         tuple(map(func, self.flatten(d[kx]))))
                elif _k == "list":
                    print("{} => {}".format(d, result))
                    self.assertEqual(tuple(self.flatten(result)),
                                     tuple(map(func, self.flatten(d))))

    def flatten(self, object):
        f = object
        if isinstance(object, (pd.DataFrame, pd.Series)):
            f = object.values.reshape(-1)
        else:
            f = np.array(object).reshape(-1)

        return f


if __name__ == "__main__":
    unittest.main()
