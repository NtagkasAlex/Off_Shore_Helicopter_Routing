import unittest
from itertools import permutations

class TSPTestCase(unittest.TestCase):
    def setUp(self):
        # Mocking problem_data with a simple distance matrix
        # 0 represents the starting point (depot)
        # dist_matrix[i][j] represents the distance from platform i to platform j
        self.mock_problem_data = type('', (), {})()  # Create a simple object to simulate problem_data
        self.mock_problem_data.distances = [
            [0, 10, 15, 20],  # Distances from depot (0) to platforms 1, 2, 3
            [10, 0, 35, 25],  # Distances from platform 1 to other platforms
            [15, 35, 0, 30],  # Distances from platform 2 to other platforms
            [20, 25, 30, 0]   # Distances from platform 3 to other platforms
        ]

        # Mocking the solver class that has solve_tsp method
        class MockSolver:
            def __init__(self, problem_data):
                self.problem_data = problem_data

            def solve_tsp(self, S, max_value):
                dist_matrix = self.problem_data.distances

                if len(S) <= 1:
                    return dist_matrix[0][S[0]] + dist_matrix[S[0]][0] if S else 0

                min_cost = max_value
                for perm in permutations(S):
                    cost = dist_matrix[0][perm[0]]  # From start to first platform
                    for i in range(1, len(perm)):
                        cost += dist_matrix[perm[i - 1]][perm[i]]
                    cost += dist_matrix[perm[-1]][0]  # From last platform back to start

                    if cost < min_cost:
                        min_cost = cost
                return min_cost

        # Create an instance of the MockSolver with mocked problem_data
        self.solver = MockSolver(self.mock_problem_data)

    def test_solve_tsp_single_platform(self):
        # Case 1: Test with only one platform in S (should return the cost to and from that platform)
        S = [1]
        max_value = float('inf')
        expected_cost = 10 + 10  # From depot (0) to platform 1 and back
        result = self.solver.solve_tsp(S, max_value)
        self.assertEqual(result, expected_cost)

    def test_solve_tsp_two_platforms(self):
        # Case 2: Test with two platforms in S
        S = [1, 2]
        max_value = float('inf')
        # Expected path: 0 -> 1 -> 2 -> 0 with cost = 10 + 35 + 15 = 60
        expected_cost = 60
        result = self.solver.solve_tsp(S, max_value)
        self.assertEqual(result, expected_cost)

    def test_solve_tsp_three_platforms(self):
        # Case 3: Test with three platforms in S
        S = [1, 2, 3]
        max_value = float('inf')
        # Minimum path: 0 -> 1 -> 3 -> 2 -> 0 with cost = 10 + 25 + 30 + 15 = 80
        expected_cost = 80
        result = self.solver.solve_tsp(S, max_value)
        self.assertEqual(result, expected_cost)

    def test_solve_tsp_no_platforms(self):
        # Case 4: Test with no platforms (should return 0)
        S = []
        max_value = float('inf')
        expected_cost = 0
        result = self.solver.solve_tsp(S, max_value)
        self.assertEqual(result, expected_cost)

if __name__ == '__main__':
    unittest.main()
