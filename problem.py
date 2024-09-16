from collections import defaultdict
import numpy as np
from utils import *
import cProfile
import pstats
import itertools
from python_tsp.exact import *

from itertools import permutations
ITERATION_LIMIT =1000
import time

MAX_COLUMNS_PER_ITERATION = 15
# class Platform:
#     def __init__(self,):

class ColumnGeneration:
    def __init__(self,problem_data:'RoutingProblem_Data'):

        W_1 = np.diag(np.maximum(np.minimum(problem_data.D[1:], problem_data.C),1))
        
        self.d_current=2*problem_data.distances[0,1:]#+problem_data.distances[1:,0]
        self.W_current=W_1
        self.D_current=problem_data.D[1:]
        
        self.problem_data=problem_data
        
        # Create the PuLP problem to maximize the objective function
        n, m = self.W_current.shape

        pulp_problem = pulp.LpProblem("Maximize_DTy", pulp.LpMaximize)
        
        self.y = [pulp.LpVariable(f'y_{i}', lowBound=None, upBound=None) for i in range(n)]
        
        
        pulp_problem += pulp.lpSum(self.D_current[i] * self.y[i] for i in range(n)), "Objective"
        W_y = self.W_current.T @ np.array(self.y)

        constraints = [(W_y[j] <= self.d_current[j], f"Constraint_{j}") for j in range(m)]

        for constraint in constraints:
            pulp_problem += constraint

        self.pulp_problem=pulp_problem
        
        self.num_vars=n
        self.num_constrains=m



    def add_column(self,W_new,d_new):
        new_constraint = pulp.lpSum(W_new[i] * self.y[i] for i in range(self.num_vars)) <= d_new

        # Add the new constraint to the LP problem
        self.pulp_problem += new_constraint
    def solve_lp(self):
        self.pulp_problem.solve(pulp.PULP_CBC_CMD(msg=False))  # Suppress the solver output
        # problem.solve()  # Suppress the solver output

        # Extract the solution as a numpy array
        y_solution = np.array([pulp.value(var) for var in self.y])
        
        return y_solution
    def simplex_step(self):

        return dual_optimal(self.d_current,self.W_current,self.D_current)

    def next_lex_subset(self,z, K, extend):
        #works#
        if len(z) > K:
            while len(z) > K:
                z.pop()  

        n = len(z)

        if extend and n < K and (n == 0 or z[n-1] < K-1):
            # Add one more item to the set
            z.append(0 if n == 0 else z[n-1] + 1)
            n += 1
        else:
            
            # Increase the current item
            z[n-1] += 1
            while z[n-1] >= K:
                
                n -= 1
                if n == 0:
                    return False
                z[n-1] += 1
        
            # print("N",n)
            z[:] = z[:n]

        return True
    
    def solve_tsp(self, S, max_value):
        """
        Solves the traveling salesman problem (TSP) for the subset of platforms S.
        Uses a dynamic programming approach with memoization (Held-Karp Algorithm).
        """

        dist_matrix = self.problem_data.distances

        if len(S) <= 1:
            return dist_matrix[0][S[0]] + dist_matrix[S[0]][0] if S else 0

        # Create a dictionary to store minimum costs of subproblems
        n = len(S)
        S_set = tuple(S)

        # Memoization table
        dp = {}

        # Initialize the distance from the start (0) to each node in S
        for i in range(n):
            dp[(1 << i, i)] = dist_matrix[0][S_set[i]]

        # Iterate over all subsets of S of increasing size
        for subset_size in range(2, n + 1):
            for subset in itertools.combinations(range(n), subset_size):
                # Generate the bitmask for this subset
                bits = sum(1 << i for i in subset)
                for j in subset:
                    prev_bits = bits & ~(1 << j)
                    min_cost = max_value
                    # Find the minimum path to reach j from any previous node
                    for k in subset:
                        if k == j:
                            continue
                        min_cost = min(min_cost, dp[(prev_bits, k)] + dist_matrix[S_set[k]][S_set[j]])
                    dp[(bits, j)] = min_cost

        # Find the minimum cost to return to the start
        final_bits = (1 << n) - 1
        min_cost = max_value
        for j in range(n):
            min_cost = min(min_cost, dp[(final_bits, j)] + dist_matrix[S_set[j]][0])

        return min_cost
    # def solve_tsp(self,S,max_value):

    #     d=self.problem_data.distances
    #     # print(d.shape)
    #     indexes=np.copy(S)
    #     indexes=np.hstack([0,indexes])
    #     indexes=indexes.reshape(-1)
    #     # print(indexes.shape)
    #     distance_matrix=d[np.ix_(indexes,indexes)]

    #     permutation, distance = solve_tsp_dynamic_programming(distance_matrix)
    #     return distance

    def column_gen_step(self):
        """
        Implements the column generation step, as outlined in the C++ code.
        This includes generating platform subsets S, solving the TSP, calculating reduced cost, and adding columns.
        """
        N = self.problem_data.N
        R = self.problem_data.R
        C = self.problem_data.C
        D = self.problem_data.D
        # Platform indices (excluding airport)
        Pindex = [i for i in range(1, N+1) if D[i] > 0]
        # Pindex = [i for i in range(1, N+1)]
        # print(Pindex)
        y=np.zeros(N+1)

        np.set_printoptions(suppress=True)

        optimal=False
        iteration=1
        # Dual variables (you would typically update these from solving the simplex)
        while not optimal and iteration < ITERATION_LIMIT:
            # print(self.d_current.shape,self.W_current.shape,self.D_current.shape)
            print(iteration)
            # sol = dual_optimal(self.d_current,self.W_current,self.D_current)
            sol=self.solve_lp()
            y[1:]=sol
            
            # sorted_indexes=np.argsort(y)[::-1]
            # print(y.shape)
            # Pindex=sorted_indexes
            # Pindex=[index for index in sorted_indexes if D[index]>0]
            # print(len(Pindex))
            Pindex.sort(key=lambda i: -y[i])
            # print(y)

            if iteration==2:
                print(y)
                print()
                print(Pindex)
                # 
                break
            print(y)
            print()
            print(Pindex)
                # 
            consider_supersets =True
            iter=0

            pi = []  # Current subset (lexicographical order)
            
            columns_added=0

            while self.next_lex_subset(pi, len(Pindex),consider_supersets) and columns_added < MAX_COLUMNS_PER_ITERATION:
                iter+=1
                # print(pi)
                # 
                consider_supersets = True

                # Construct subset S based on pi
                S = [Pindex[i] for i in pi]
                # print(columns_added)
                # Solve TSP for subset S               
                # print(len(S))
                d_S = self.solve_tsp(S, R + 0.1)
                # print("DS")
                # print(d_S)
                # If the TSP tour length exceeds R, exclude S and its supersets
                if d_S > R:
                    # print("s")
                    consider_supersets = False
                    continue

                # Calculate reduced cost
                c = d_S
                C_remaining = C
                sum_D = 0
                w_star = np.zeros(N)

                for j in range(len(S)):
                    i=S[j]
                    w = min(C_remaining, D[i])
                    w_star[i-1]=w
                    C_remaining -= w
                    c -= w * y[i]  # Subtract the dual value
                    sum_D += D[i]


                # print(C_remaining)
                # print(w)
                # print(c)
                # print(sum_D)
                # break

                # If D exceed capacity, skip supersets
                if sum_D >= C:
                    consider_supersets = False

                # If reduced cost is negative, add the column
                if c < -1e-5:
                    
                    # Simulate adding a column by expanding W_current
                    
                    # print(S)
                    
                    # self.W_current = np.hstack((self.W_current, w_star.reshape((-1,1))))  

                    # self.d_current= np.hstack([self.d_current,d_S])

                    self.add_column(w_star,d_S)
                    columns_added += 1
                
                    # print(c)
            print("Collumns added:",columns_added)
            optimal = (columns_added == 0)
            # print(optimal)
            iteration+=1
            if (optimal):
                print("Found Optimal")
            
class RoutingProblem_Data:
    def __init__(self,platform_filename,demand_filename):
        N,R,C,platforms=self.extract_platforms_from_file(platform_filename)
        D=self.extract_demand_from_fime(demand_filename)

        self.N=N
        self.R=R
        self.C=C
        self.D=D
        
        self.airport=np.array([0,0])
        
        
        self.platforms=dict()   
        self.platforms[0]=np.array([0,0])

        for ind,x,y in platforms:
            self.platforms[ind]=np.array([x,y])
            
        distances = np.zeros((N+1, N+1))
        for i in range(N+1):
            for j in range(i+1, N+1):
                # Calculate Euclidean distance between platform i and platform j
                dist = np.linalg.norm(self.platforms[i] - self.platforms[j])
                distances[i, j] = dist
                distances[j, i] = dist  # Distance is symmetric
            
        self.distances=distances



    def extract_platforms_from_file(self,filename):
        with open(filename, 'r') as file:
            # Read first three lines for N, R, and C
            N = int(file.readline().strip())  # Number of platforms
            R = float(file.readline().strip())  # Helicopter range
            C = int(file.readline().strip())  # Helicopter capacity
            
            # Initialize list for storing platform information
            platforms = []

            # Loop through the next N lines to extract platform data
            for _ in range(N):
                line = file.readline().strip()
                platform_data = line.split()  # Split the line into components
                platform_id = int(platform_data[0])  # Platform number
                x_coord = float(platform_data[1])    # x-coordinate
                y_coord = float(platform_data[2])    # y-coordinate
                platforms.append((platform_id, x_coord, y_coord))
        
        return N, R, C, platforms
    def extract_demand_from_fime(self,filename):
        D = []
        with open(filename, 'r') as file1:
            for line in file1:

                numbers = line.strip().split() 

                first_num = numbers[0]  
                second_num = numbers[1]  
                D.append(int(second_num))  # Append both numbers as a tuple
        D.insert(0,0)
        return np.array(D)


if __name__=="__main__":
        
    # Example usage
    platform_filename = 'data/platform.txt'  # Replace with the actual filename
    demand_filename = 'data/demand-1.txt'  # Replace with the actual filename
    # platform_filename = 'data/platform-0.txt'  # Replace with the actual filename
    # demand_filename = 'data/demand-0.txt'  # Replace with the actual filename
    rp_data=RoutingProblem_Data(platform_filename,demand_filename)
    cg=ColumnGeneration(rp_data)
    # print(cg.simplex_step())
    # cg.column_gen_step()
    # Example profiling
   # Manually create a profiler
    profiler = cProfile.Profile()
    profiler.enable()

    # Run the function you want to profile
    cg.column_gen_step()

    # Disable the profiler
    profiler.disable()

    # Write results to the file
    with open('profile_results.txt', 'w') as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.strip_dirs().sort_stats('time').print_stats()

    # Print confirmation
    print("Profile results written to 'profile_results.txt'")
    p=[]
    flag=True
    count=0
    # while cg.next_lex_subset(p,3,flag):
    #     count+=1
    #     flag=True
    #     if count==2:
    #         flag=False
    #     print(p)
    # cg.next_lex_subset(p,3,True)
