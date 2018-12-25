import numpy as np
import unittest
import chariot.transformer.formatter as fmt


class TestGenerator(unittest.TestCase):

    def test_shift_generator(self):
        g = fmt.ShiftGenerator(shift=1)
        seq = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        _from, _to = g.generate(seq, 0, 3)
        self.assertEqual(_from.tolist(), seq[0:3].tolist())
        self.assertEqual(_to.tolist(), seq[1:4].tolist())

    def test_shuffle_generator(self):
        g = fmt.ShuffleGenerator()
        seq = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        _from, _to = g.generate(seq, 0, 3)
        self.assertEqual(_from.tolist(), seq[0:3].tolist())
        self.assertNotEqual(_to.tolist(), seq[1:4].tolist())


if __name__ == "__main__":
    unittest.main()
