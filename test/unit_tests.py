import unittest
import linalg

class TestVectorModule(unittest.TestCase):
    # helper functions
    def assertVectorAlmostEqual(self, vector1, vector2, places=2):
        self.assertEqual(len(vector1), len(vector2))
        for i in range(len(vector1)):
            self.assertAlmostEqual(vector1[i], vector2[i], places=places)

    # add(vector1, vector2)
    def test_addition_invalid(self):
        self.assertRaises(ValueError, lambda: linalg.add([], [1, 2, 3]))
        self.assertRaises(ValueError, lambda: linalg.add([1, 2, 3], []))
    def test_addition(self):
        self.assertEqual(linalg.add([], []), [])
        self.assertEqual(linalg.add([1], [1]), [2])
        self.assertEqual(linalg.add([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]), [6, 6, 6, 6, 6])

    # sub(vector1, vector2)
    def test_subtraction_invalid(self):
        self.assertRaises(ValueError, lambda: linalg.sub([], [1, 2, 3]))
        self.assertRaises(ValueError, lambda: linalg.sub([1, 2, 3], []))
    def test_subtraction(self):
        self.assertEqual(linalg.sub([], []), [])
        self.assertEqual(linalg.sub([1], [1]), [0])
        self.assertEqual(linalg.sub([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]), [-4, -2, 0, 2, 4])

    # dot_product(vector1, vector2)
    def test_dot_product_invalid(self):
        self.assertRaises(ValueError, lambda: linalg.dot_product([], [1, 2, 3]))
        self.assertRaises(ValueError, lambda: linalg.dot_product([1, 2, 3], []))
    def test_dot_product(self):
        self.assertEqual(linalg.dot_product([], []), 0)
        self.assertEqual(linalg.dot_product([1, 0, 3], [0, 2, 0]), 0)
        self.assertEqual(linalg.dot_product([1, 2, 3], [4, 5, 6]), 32)
        self.assertEqual(linalg.dot_product([1, 2, 3], [-4, -5, -6]), -32)

    # scale(vector, scalar)
    def test_scale(self):
        self.assertEqual(linalg.scale([], 1), [])
        self.assertEqual(linalg.scale([1], 10), [10])
        self.assertEqual(linalg.scale([1, 2, 3], 0), [0, 0, 0])
        self.assertEqual(linalg.scale([1, 2, 3, 4, 5, 6], 25), [25, 50, 75, 100, 125, 150])

    # magnitude(vector)
    def test_magnitude(self):
        self.assertEqual(linalg.magnitude([]), 0)
        self.assertEqual(linalg.magnitude([1]), 1)
        self.assertAlmostEqual(linalg.magnitude([1, 2, 3]), 3.74166, places=2)
        self.assertAlmostEqual(linalg.magnitude([1, 4, 9, 16]), 18.81, places=2)
        self.assertAlmostEqual(linalg.magnitude([2, 2, 88, 25, 71, 57, 16, 90, 89, 57]), 190.245, places=2)
        self.assertAlmostEqual(linalg.magnitude([993, 458, 883, 887, 29, 212, 677, 99, 264, 287, 923, 102, 170, 438, 780, 897, 317, 646, 525, 207, 14, 224, 897, 447, 659, 209, 577, 27, 31, 552, 385, 870, 766, 134, 351, 323, 750, 817, 480, 815, 178, 450, 620, 876, 398, 347, 102, 353, 115, 801, 40, 572, 552, 202, 658, 432, 461, 553, 64, 387, 987, 645, 992, 171, 550, 84, 136, 817, 194, 956, 490, 817, 288, 201, 470, 503, 31, 179, 614, 122, 79, 16, 551, 749, 571, 998, 64, 876, 104, 478, 461, 577, 93, 838, 997, 91, 201, 700, 333, 190]), 5443.86, places=2)

    # normalize(vector)
    def test_normalize(self):
        input = [1, 2, 3]
        output = linalg.normalize(input)
        expected = [0.267, 0.534, 0.802]
        self.assertVectorAlmostEqual(output, expected)


class TestMatrixModule(unittest.TestCase):
    # determinant(matrix)
    def test_determinant(self):
        self.assertEqual(linalg.determinant([[3.14]]), 3.14)
        self.assertEqual(linalg.determinant([[8, 6], [3, 4]]), 14)
        self.assertEqual(linalg.determinant([[3, 1, 1], [4, -2, 5], [2, 8, 7]]), -144)
        self.assertAlmostEqual(linalg.determinant([
            [9.8656, 10.0306, 0.3472, 11.0069, 7.3139, 4.4606, 4.2002],
            [2.0655, 14.5462, 0.9602, 13.7961, 2.8015, 9.0956, 5.0173],
            [5.2817, 14.3850, 9.1309, 10.1272, 5.2193, 2.4805, 2.9069],
            [6.0149, 14.8712, 7.1108, 3.9310, 10.5918, 0.5560, 11.3649],
            [4.4826, 7.7488, 12.8341, 9.5523, 6.5164, 9.0212, 9.4849],
            [6.6361, 12.5463, 2.7712, 1.9945, 7.0052, 7.3419, 12.4871],
            [4.0405, 3.5168, 1.0304, 9.5555, 2.0069, 12.1179, 1.0339]]), 3710595.29, places=2)
    # det(I) = 1
    def test_determinant_identity(self):
        self.assertEqual(linalg.determinant([[1]]), 1)
        self.assertEqual(linalg.determinant([[1, 0], [0, 1]]), 1)
        self.assertEqual(linalg.determinant([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), 1)

    # det with zero row/col = 0
    def test_determinant_2_zero_row(self):
        self.assertEqual(linalg.determinant([[2, 2], [0, 0]]), 0)
        self.assertEqual(linalg.determinant([[2, 2, 2, 2, 2], [0, 0, 0, 0, 0], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2]]), 0)
    def test_determinant_2_zero_col(self):
        self.assertEqual(linalg.determinant([[2, 0], [2, 0]]), 0)

    # det(L) = det(U) = diagonal product
    def test_determinant_triangular(self):
        self.assertEqual(linalg.determinant([[3, 1, 1], [0, 2, 5], [0, 0, 4]]), 24)
        self.assertEqual(linalg.determinant([[4, 0, 0], [5, 2, 0], [1, 1, 3]]), 24)

if __name__ == '__main__':
    unittest.main()
