import numpy as np
import cProfile
import pstats
import itertools
import copy
import matplotlib.pyplot as plt
import pulp
import time
from typing import List, Tuple
import sys

ITERATION_LIMIT =1000
MAX_COLUMNS_PER_ITERATION = 15


def print_flights_nicely(flights, precision=4):
    np.set_printoptions(precision=precision)
    print()
    print()
    print(f"{'x_j':<20}{'Platforms':<20}{'Crew Exchanges':<30}{'Distance':<10}")
    print("-" * 90)
    for flight in flights:
        x, platforms, crew_exchanges, distance = flight
        x_str=f"{x:.3f}"
        platforms_str = ', '.join([f"{int(platform)}" for platform in platforms])  
        crew_exchanges_str = ', '.join([f"{int(crew)}" for crew in crew_exchanges])  
        print(f"{x_str:<20}{platforms_str:<20}{crew_exchanges_str:<30}{distance:.3f}")  


def get_flights(flights_input):
    
    flights=[]
    for flight in flights_input:
        x,w,d=flight
        if x>1e-5:
            platforms=[]
            crew_exchanges=[]
            for j,weight in enumerate(w):
                if weight>1e-5:
                    platforms.append(j+1)
                    crew_exchanges.append(weight)
            
            flights.append([x,platforms,crew_exchanges,d])

    # print_flights_nicely(self.flights)
    
    return flights

def get_obj_value(flights):
    z=0
    for flight in flights:
        x,_,_,d=flight
        z+=x*d

    return z
class TSPSolver:
    def __init__(self):
        self.tsp_cache = {}
        self.tsp_count = 0
        self.last_tsp_report = 0
        self.tsp_cache_hit = 0
        self.tsp_solve_time = 0
        self.tsp_cache_time = 0

    def solve_tsp(self, S: List[int], d: List[List[float]], max_value: float) -> float:
        """
        Calculate the shortest traveling salesman tour starting and ending at the airport,
        and visiting all platforms in S. The total distance must be less than or equal to max_value.
        """
        self.tsp_count += 1
        n = len(S)
        min_distance = max_value

        S_sorted = sorted(S)

        tsp_cache_key = tuple(S_sorted)

        if tsp_cache_key in self.tsp_cache:
            self.tsp_cache_hit += 1
            return self.tsp_cache[tsp_cache_key]

        min_way_back = min(d[0][S[i]] for i in range(n))

        for perm in itertools.permutations(S_sorted):


            #  TSP tours are symmetric
            if perm[0] > perm[-1]:

                continue

            perm_distance = d[0][perm[0]]  
            istar = -1

            for i in range(1, n):
                perm_distance += d[perm[i - 1]][perm[i]] 
                if perm_distance + min_way_back >= min_distance:
                    istar = i
                    break

            if istar >= 0:
                continue

            perm_distance += d[perm[-1]][0]

            if perm_distance < min_distance:
                min_distance = perm_distance

        self.tsp_cache[tsp_cache_key] = min_distance

        return min_distance

class ColumnGeneration:
    def __init__(self,d_current,W_current,D_current,problem_data:'RoutingProblem_Data',print_progress=True):
        
        self.d_current=d_current
        self.W_current=W_current
        self.D_current=D_current
        
        self.problem_data=copy.copy(problem_data)
        
        n, m = self.W_current.shape

        dual_problem = pulp.LpProblem("Maximize_DTy", pulp.LpMaximize)
        
        self.y = [pulp.LpVariable(f'y_{i}', lowBound=None, upBound=None) for i in range(n)]
        
        
        dual_problem += pulp.lpSum(self.D_current[i] * self.y[i] for i in range(n)), "Objective"
        W_y = self.W_current.T @ np.array(self.y)
        
        constraints = [(W_y[j] <= self.d_current[j], f"Constraint_{j}") for j in range(m)]

        for constraint in constraints:
            dual_problem += constraint

        self.dual_problem=dual_problem
        
        self.num_vars=n
        self.num_constrains=m

        self.solver = TSPSolver()

        self.flights=[]

        self.print_progress=print_progress

    def add_column(self,W_new,d_new):

        self.W_current = np.hstack((self.W_current, W_new.reshape((-1,1))))  

        self.d_current= np.hstack([self.d_current,d_new])

        new_constraint = pulp.lpSum(W_new[i] * self.y[i] for i in range(self.num_vars)) <= d_new

        self.dual_problem += new_constraint

    def solve_primal(self):
        W=self.W_current
        d=self.d_current
        D=self.D_current
        
        n_vars = W.shape[1]

        lp_problem = pulp.LpProblem("Minimize_dTx", pulp.LpMinimize)

        
        x_vars = [pulp.LpVariable(f"x{i+1}", lowBound=0) for i in range(n_vars)]
        
        lp_problem += pulp.lpSum([d[i] * x_vars[i] for i in range(n_vars)]), "Objective"

        for i in range(W.shape[0]):
            lp_problem += pulp.lpSum([W[i, j] * x_vars[j] for j in range(n_vars)]) == D[i], f"Constraint_{i+1}"

        lp_problem.solve(pulp.PULP_CBC_CMD(msg=False))
    
        self.primal_problem=lp_problem

        x_solution = np.array([pulp.value(var) for var in x_vars])
        # print(f"tive value: {pulp.value(lp_problem.objective)}")

        return x_solution
    
    def solve_dual(self):
        self.dual_problem.solve(pulp.PULP_CBC_CMD(msg=False))  
        # problem.solve()  

        y_solution = np.array([pulp.value(var) for var in self.y])
        
        return y_solution
    def print_primal(self):
        print("Objective Function:")
        print(self.primal_problem.objective)

        print("\nConstraints:")
        for name, constraint in self.primal_problem.constraints.items():
            print(f"{name}: {constraint}")
    def print_lp(self):
        print("Objective Function:")
        print(self.dual_problem.objective)

        print("\nConstraints:")
        for name, constraint in self.dual_problem.constraints.items():
            print(f"{name}: {constraint}")



    def next_lex_subset(self,z, K, extend):
        #works#
        if len(z) > K:
            while len(z) > K:
                z.pop()  

        n = len(z)

        if extend and n < K and (n == 0 or z[n-1] < K-1):
            z.append(0 if n == 0 else z[n-1] + 1)
            n += 1
        else:
            
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
        

        dist_matrix = self.problem_data.distances

        cost=self.solver.solve_tsp(S, dist_matrix, max_value)

        return cost

    def get_flights(self):
        x_total=self.solve_primal()
        self.flights=[]
        for i,x in enumerate(x_total):
            if x>1e-5:
                platforms=[]
                crew_exchanges=[]
                for j,weight in enumerate(self.W_current[:,i]):
                    if weight>1e-5:
                        platforms.append(j+1)
                        crew_exchanges.append(weight)
                
                d=self.d_current[i]
                self.flights.append([x,platforms,crew_exchanges,d])

        if self.print_progress:print_flights_nicely(self.flights)
        
        return self.flights
                
                
    def get_objective_value(self):
        # print(f"tive value: {pulp.value(self.primal_problem.objective)}")
        return pulp.value(self.primal_problem.objective)
    def column_gen_step(self):
        
        N = self.problem_data.N
        R = self.problem_data.R
        C = self.problem_data.C
        D = self.problem_data.D

        Pindex = [i for i in range(1, N+1) if D[i] > 0]
        # Pindex = [i for i in range(1, N+1)]
        # print(Pindex)
        y=np.zeros(N+1)

        np.set_printoptions(suppress=True)
        
        optimal=False
        iteration=1
        while not optimal and iteration < ITERATION_LIMIT:

            if self.print_progress:print(iteration)

            sol=self.solve_dual()
            y[1:]=sol

            Pindex.sort(key=lambda i: -y[i])

           
            consider_lex_supersets =True
            iter=0

            pi = []  
            
            columns_added=0

            while self.next_lex_subset(pi, len(Pindex),consider_lex_supersets) and columns_added < MAX_COLUMNS_PER_ITERATION:
                iter+=1
                
                consider_lex_supersets = True

                S = [Pindex[i] for i in pi]
                
                d_S = self.solve_tsp(S, R + 0.1)
               
                if d_S > R:
                    consider_lex_supersets = False
                    continue

                c = d_S
                C_remaining = C
                sum_D = 0
                w_star = np.zeros(N)

                for j in range(len(S)):
                    i=S[j]
                    w = min(C_remaining, D[i])
                    w_star[i-1]=w
                    C_remaining -= w
                    c -= w * y[i]  
                    sum_D += D[i]


                if sum_D >= C:
                    consider_lex_supersets = False

                if c < -1e-5:
                    
                    self.add_column(w_star,d_S)

                    columns_added += 1

                    # print(c)
            if self.print_progress:print("Collumns added:",columns_added)
            optimal = (columns_added == 0)

            iteration+=1
            
        if (optimal):
            if self.print_progress:print("Found Optimal")
            self.get_flights()
            
            return self.get_objective_value()
                
            
class RoutingProblem_Data:
    def __init__(self,platform_filename,demand_filename,N=None):
        N,R,C,platforms=self.extract_platforms_from_file(platform_filename,N)

        D=self.extract_demand_from_fime(demand_filename)
        if N is not None:
            D=D[:N+1]
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

                dist = np.linalg.norm(self.platforms[i] - self.platforms[j])
                distances[i, j] = dist
                distances[j, i] = dist  
            
        self.distances=distances



    def extract_platforms_from_file(self,filename,N=None):
        if N is None:
            with open(filename, 'r') as file:
                N = int(file.readline().strip())  
                R = float(file.readline().strip())  
                C = int(file.readline().strip())  
                
                platforms = []

                for _ in range(N):
                    line = file.readline().strip()
                    platform_data = line.split()  
                    platform_id = int(platform_data[0])  
                    x_coord = float(platform_data[1])    
                    y_coord = float(platform_data[2])    
                    platforms.append((platform_id, x_coord, y_coord))
            
            return N, R, C, platforms
        else:
            with open(filename, 'r') as file:
                _ = int(file.readline().strip())  
                R = float(file.readline().strip())  
                C = int(file.readline().strip())  
                
                platforms = []

                for _ in range(N):
                    line = file.readline().strip()
                    platform_data = line.split()  
                    platform_id = int(platform_data[0])  
                    x_coord = float(platform_data[1])    
                    y_coord = float(platform_data[2])    
                    platforms.append((platform_id, x_coord, y_coord))
            
            return N, R, C, platforms
    def extract_demand_from_fime(self,filename):
        D = []
        with open(filename, 'r') as file1:
            for line in file1:

                numbers = line.strip().split() 
                second_num = numbers[1]  
                D.append((second_num))  

        D.insert(0,0)
        return np.array(D,dtype=np.float64)
    
    def plot_flights(self,flights):
        platforms_dict=self.platforms

        plt.figure(figsize=(10, 8))

        airport_coords = platforms_dict[0]
        plt.scatter(airport_coords[0], airport_coords[1], color='red', marker='o', s=100, label='Airport (Platform 0)')

        for ind, coord in platforms_dict.items():
            if ind != 0:
                plt.scatter(coord[0], coord[1], color='blue', marker='o')
                plt.text(coord[0], coord[1], f'{ind}', fontsize=12, ha='right')

        
        for path in flights:
            if len(path) > 0:
                current_platform = 0
                flight_color = list(np.random.rand(3))  
                for platform in path:
                    plt.arrow(platforms_dict[current_platform][0], platforms_dict[current_platform][1],
                            platforms_dict[platform][0] - platforms_dict[current_platform][0],
                            platforms_dict[platform][1] - platforms_dict[current_platform][1],
                            head_width=1, head_length=1, fc=flight_color, ec=flight_color)

                plt.arrow(platforms_dict[current_platform][0], platforms_dict[current_platform][1],
                        airport_coords[0] - platforms_dict[current_platform][0],
                        airport_coords[1] - platforms_dict[current_platform][1],
                        head_width=1, head_length=1, fc=flight_color, ec=flight_color)

        # Add labels and grid
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Flights from the Airport to Platforms and Back')
        plt.grid(True)
        plt.legend()

        # Show plot
        plt.show()
    def plot_platforms(self):

        platforms_dict=self.platforms
        x_coords = [platforms_dict[ind][0] for ind in platforms_dict]
        y_coords = [platforms_dict[ind][1] for ind in platforms_dict]

        plt.figure(figsize=(8, 6))
        plt.scatter(x_coords, y_coords, color='blue', marker='o')

        for ind in platforms_dict:
            plt.text(platforms_dict[ind][0], platforms_dict[ind][1], f'{ind}', fontsize=12, ha='right')

        # Label axes
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('2D Plot of Platforms')

        # Show plot
        plt.grid(True)
        plt.show()

def round_off_procidure(data,print_progress=True):
    data=copy.copy(data)
    d_current = 2*data.distances[0,1:]
    W_current = np.diag(np.maximum(np.minimum(data.D[1:], data.C),1))
    D_current = data.D[1:]

    sumD=np.sum(D_current)

    # seed = 50
    seed=np.random.randint(0,1000)
    np.random.seed(seed)
    flights=[]

    while sumD>0:
        assert(d_current.shape[0]==W_current.shape[1] and D_current.shape[0]==W_current.shape[0])
        cg=ColumnGeneration(d_current,W_current,D_current,data,print_progress)
        
        z=cg.column_gen_step()
        # print("Optimal Obj value: ", z)
        # print(W_current.shape,d_current.shape,D_current.shape) 
        x=cg.solve_primal()
        
        W_current=cg.W_current
        D_current=cg.D_current
        d_current=cg.d_current
        if np.all(np.abs(x - np.round(x)) < 1e-6) :
            for i,x_i in enumerate(x):
                if x_i>1e-5:
                    flights.append([x_i,W_current[:,i],d_current[i]])
            break

        valid_indices = np.where(x > 0)[0]

        j = np.random.choice(valid_indices)

        x_j=x[j]

        
        
        x_j=1 if x_j<1 else np.floor(x_j)

        flights.append([x_j,W_current[:,j],d_current[j]])

        D_current-=W_current[:,j]*x_j
        
        sumD-=np.sum(W_current[:,j]*x_j)
        


        condition = W_current > D_current.reshape((-1,1))
        
        columns_to_keep = ~np.any(condition, axis=0)

        W_current = W_current[:, columns_to_keep]

        d_current=d_current[columns_to_keep]

        W_current=np.hstack([W_current,np.diag(np.maximum(np.minimum(D_current, data.C),1))])

        d_current=np.hstack([d_current,2*data.distances[0,1:]])

    return flights
if __name__=="__main__":
        
    # Example usage
    platform_filename = 'data/platform.txt'  
    demand_filename = 'data/demand-1.txt'    
    # platform_filename = 'data/platform-0.txt'  
    # demand_filename = 'data/demand-0.txt'      
    # platform_filename = 'data/platform-15.txt' 
    # demand_filename = 'data/demand-15.txt'
    # platform_filename=sys.argv[1]
    # demand_filename=sys.argv[2]

    #########################################
    #For Column Generation

    N_platforms=20
    n_trainings=1
    percentages=[]
    times=[]
    for i in range(n_trainings):
        t1=time.time()
        rp_data=RoutingProblem_Data(platform_filename,demand_filename,N_platforms)
        d_current = 2*rp_data.distances[0,1:]
        W_current = np.diag(np.maximum(np.minimum(rp_data.D[1:], rp_data.C),1))
        D_current = rp_data.D[1:]
        cg=ColumnGeneration(d_current,W_current,D_current,rp_data,print_progress=True)
        z=cg.column_gen_step()
        print(f"Objective value: {z}")
        t2=time.time()
        times.append(t2-t1)
    # print(cg.solver.tsp_cache_hit)
    # # print(cg.solver.tsp_cache)
    # print(cg.solver.last_tsp_report)
    # print(cg.solver.tsp_cache_time)
    # print(cg.solver.tsp_count)
    # print(cg.solver.tsp_solve_time)

    #########################################
    #For Round Off

    # N_platforms=51
    # n_trainings=1
    # percentages=[]
    # times=[]
    # for i in range(n_trainings):
    #     t1=time.time()
    #     rp_data=RoutingProblem_Data(platform_filename,demand_filename,N_platforms)

    #     d_current = 2*rp_data.distances[0,1:]
    #     W_current = np.diag(np.maximum(np.minimum(rp_data.D[1:], rp_data.C),1))
    #     D_current = rp_data.D[1:]
    #     print(d_current.shape,W_current.shape,D_current.shape)
    #     print(N_platforms)
    #     cg=ColumnGeneration(d_current,W_current,D_current,rp_data,print_progress=False)
        
    #     z_opt=cg.column_gen_step()

    #     f=round_off_procidure(rp_data,print_progress=True)
        
    #     f=get_flights(f)
        
    #     z_round=get_obj_value(f)
    #     t2=time.time()


    #     # print_flights_nicely(f)
    #     percentage=100.0 * (z_round - z_opt) / z_opt
    #     print()
    #     print(z_round)
    #     print("Rounded solution is  {:.3f} % more expensive than the optimal solution.".format(percentage))
    #     percentages.append(percentage)
    #     times.append(t2-t1)
        
    ####################################
    #For Saving
    # np.save("results-"+str(N_platforms)+".npy",np.array(percentages))
    # np.save("times_cg-"+str(N_platforms)+".npy",np.array(times))

    # rp_data=RoutingProblem_Data(platform_filename,demand_filename)

    # print(rp_data.D[1:])