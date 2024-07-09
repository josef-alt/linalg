import unittest
import linalg

class TestLinAlg(unittest.TestCase):

    # sum_as_string(a, b)
    def test_sum_as_string_zero(self):
        self.assertEqual(linalg.sum_as_string(0, 0), "0")
    def test_sum_as_string_pos(self):
        self.assertEqual(linalg.sum_as_string(1, 10), "11")

    # list_to_string(list)
    def test_list_to_string_empty(self):
        self.assertEqual(linalg.list_to_string([]), "[]")
    def test_list_to_string(self):
        self.assertEqual(linalg.list_to_string([1, 2, 3]), "[1, 2, 3]")

    # scale_list(list, scalar)
    def test_scale_list_empty(self):
        test = []
        linalg.scale_list(test, 10)
        self.assertEqual(test, [])
    def test_scale_list_zero(self):
        test = [1, 2, 3]
        linalg.scale_list(test, 0)
        self.assertEqual(test, [0, 0, 0])
    def test_scale_list(self):
        test = [1, 2, 3]
        linalg.scale_list(test, 2)
        self.assertEqual(test, [2, 4, 6])

    # list_dot_product(list1, list2)
    def test_list_dot_product_invalid(self):
        self.assertRaises(ValueError, lambda: linalg.list_dot_product([], [1, 2, 3]))
    def test_list_dot_product_empty(self):
        self.assertEqual(linalg.list_dot_product([], []), 0)
    def test_list_dot_product_zero(self):
        self.assertEqual(linalg.list_dot_product([1, 0, 3], [0, 2, 0]), 0)
    def test_list_dot_product_pos(self):
        self.assertEqual(linalg.list_dot_product([1, 2, 3], [4, 5, 6]), 32)
    def test_list_dot_product_neg(self):
        self.assertEqual(linalg.list_dot_product([1, 2, 3], [-4, -5, -6]), -32)

    # determinant(matrix)
    def test_determinant(self):
        self.assertEqual(linalg.determinant([[3.14]]), 3.14)
        self.assertEqual(linalg.determinant([[8, 6], [3, 4]]), 14)
        self.assertEqual(linalg.determinant([[3, 1, 1], [4, -2, 5], [2, 8, 7]]), -144)
        self.assertEqual(linalg.determinant([
            [9.8656, 10.0306, 0.3472, 11.0069, 7.3139, 4.4606, 4.2002],
            [2.0655, 14.5462, 0.9602, 13.7961, 2.8015, 9.0956, 5.0173],
            [5.2817, 14.3850, 9.1309, 10.1272, 5.2193, 2.4805, 2.9069],
            [6.0149, 14.8712, 7.1108, 3.9310, 10.5918, 0.5560, 11.3649],
            [4.4826, 7.7488, 12.8341, 9.5523, 6.5164, 9.0212, 9.4849],
            [6.6361, 12.5463, 2.7712, 1.9945, 7.0052, 7.3419, 12.4871],
            [4.0405, 3.5168, 1.0304, 9.5555, 2.0069, 12.1179, 1.0339]]), 3.7106e6)
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
