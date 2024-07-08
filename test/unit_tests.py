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
    def test_determinant_1(self):
        self.assertEqual(linalg.determinant([[1]]), 1)
    
    # det(I) = 1
    def test_determinant_2_identity(self):
        self.assertEqual(linalg.determinant([[1, 0], [0, 1]]), 1)

    # det with zero row/col = 0
    def test_determinant_2_zero_row(self):
        self.assertEqual(linalg.determinant([[2, 2], [0, 0]]), 0)
    def test_determinant_2_zero_col(self):
        self.assertEqual(linalg.determinant([[2, 0], [2, 0]]), 0)
    def test_determinant_2_1(self):
        self.assertEqual(linalg.determinant([[8, 6], [3, 4]]), 14)

    # det(L) = det(U) = diagonal product
    def test_determinant_3_upper_triangular(self):
        self.assertEqual(linalg.determinant([[3, 1, 1], [0, 2, 5], [0, 0, 4]]), 24)
    def test_determinant_3_lower_triangular(self):
        self.assertEqual(linalg.determinant([[4, 0, 0], [5, 2, 0], [1, 1, 3]]), 24)
    def test_determinant_3_1(self):
        self.assertEqual(linalg.determinant([[3, 1, 1], [4, -2, 5], [2, 8, 7]]), -144)

if __name__ == '__main__':
    unittest.main()
