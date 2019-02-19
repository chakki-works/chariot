import numpy as np
import pandas as pd
import unittest
import chariot.transformer.formatter as fmt


class TestFormatter(unittest.TestCase):

    def test_categorical_label(self):
        formatter = fmt.CategoricalLabel(num_class=5)
        column = np.array([1, 3, 2])
        categorical = formatter.transform(column)
        for i, n in enumerate(column):
            self.assertEqual(n, np.argmax(categorical[i, :]))

    def test_padding(self):
        length = 7
        bos = 10
        eos = 11
        formatter = fmt.Padding(padding=0, length=length,
                                begin_of_sequence=bos, end_of_sequence=eos)
        data = [
            [1, 2],
            [3, 4, 5],
            [1, 2, 3, 4, 5]
        ]

        padded = formatter.transform(data)
        for d, p in zip(data, padded):
            self.assertEqual(len(p), length)
            self.assertEqual(p[0], bos)
            padding_size = length - len(d) - len([bos, eos])
            self.assertEqual(len(["p" for x in p if x == 0]), padding_size)
            self.assertEqual(tuple(d), tuple(formatter.inverse_transform([p])[0]))

    def test_padding_from_series(self):
        formatter = fmt.Padding(padding=0)
        data = [
            [1, 2],
            [3, 4, 5],
            [1, 2, 3, 4, 5]
        ]

        padded = formatter.transform(pd.Series(data))
        max_length = 5
        for d, p in zip(data, padded):
            print(p)
            self.assertEqual(len(p), max_length)


if __name__ == "__main__":
    unittest.main()
