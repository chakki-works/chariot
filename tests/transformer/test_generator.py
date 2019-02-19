import numpy as np
import unittest
import chariot.transformer.generator as gen


class TestGenerator(unittest.TestCase):

    def test_shift_target(self):
        g = gen.ShiftedTarget(shift=1)
        seq = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        _from, _to = g.generate(seq, 0, 3)
        self.assertEqual(_from.tolist(), seq[0:3].tolist())
        self.assertEqual(_to.tolist(), seq[1:4].tolist())

    def test_shuffle_target(self):
        g = gen.ShuffledTarget()
        seq = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        _from, _to = g.generate(seq, 0, 3)
        self.assertEqual(_from.tolist(), seq[0:3].tolist())
        self.assertNotEqual(_to.tolist(), seq[1:4].tolist())

    def test_shuffle_source(self):
        g = gen.ShuffledSource()
        seq = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        _from, _to = g.generate(seq, 0, 3)
        self.assertEqual(_to.tolist(), seq[0:3].tolist())
        self.assertNotEqual(_from.tolist(), seq[1:4].tolist())


if __name__ == "__main__":
    unittest.main()
