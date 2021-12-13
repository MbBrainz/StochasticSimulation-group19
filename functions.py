# %%
import numpy as np
import math
from random import seed
from random import random
from numba import jit

# %%
def initialParameter():
    # custom function initParameter():
    # Initial parameter for simulated annealing algorithm
    t_start = 100.0
    t_final  = 1
    nMarkov = 1000
    cooling_factor    = 0.98

    return t_start,t_final,cooling_factor,nMarkov

def read_tsp_file(tsp_file):
    coords = []
    datafile = open(tsp_file,"r")

    while(1):
        line = datafile.readline()
        if "NODE_COORD_SECTION" in line:
            break

    while(1):
        line = datafile.readline()
        if "EOF" in line:
            break
        lineList = line.split()

        tupleNode = (int(lineList[0]), float(lineList[1]),float(lineList[2]))
        coords.append(tupleNode)
    datafile.close()
    return coords


# Euc distance
def distance(n1, n2):
    return math.sqrt((n1[0] - n2[0]) ** 2 + (n1[1] - n2[1]) ** 2)




# create upper triangular matrix
def distance_matrix(coords):
    num_nodes = len(coords)
    dis_matrix = np.empty([num_nodes,num_nodes])
    dis_matrix.fill(0)

    for i in range(num_nodes-1):
        tupleNode1 = coords[i]
        # node number
        node_1 = tupleNode1[0]
        # x and y coordinates
        n1 = (tupleNode1[1],tupleNode1[2])
        j = i+1
        while(j<num_nodes):
            tupleNode2 = coords[j]
            n2 = (tupleNode2[1],tupleNode2[2])
            node_2 = tupleNode2[0]
            #print(n1)
            #print(n2)
            dist_n1_n2 = distance(n1,n2)
            #print(dist_n1_n2)
            #print("\n")

            # row - node 1 and col - node 2

            dis_matrix[node_1 - 1][node_2 - 1] = dist_n1_n2
            dis_matrix[node_2 - 1][node_1 - 1] = dis_matrix[node_1 - 1][node_2 - 1]
            j = j+1
            #print(dist_n1_n2)


    return dis_matrix

def total_dist(route,distance_matrix):



    num_nodes = len(route)
    total_dis = 0
    for i in range(num_nodes-1):
        n1 = route[i]
        n2 = route[i+1]
        total_dis += distance_matrix[n1][n2]
    return total_dis

# Use swapping

def generate_i_j(num_cities):

    # swap the position of the two Cities

    i = np.random.randint(low=1, high=num_cities-1,dtype=int)
    j = np.random.randint(low=1, high=num_cities-1,dtype=int)

    # generate j till i!=j
    while i==j:
        j = np.random.randint(low=1, high=num_cities-1,dtype=int)

    i, j = min(i,j), max(i,j)

    return i,j

# BR algorithm
def two_opt_swap(route, i, k):
    new_route = np.copy(route)
    new_route[i:k+1] = route[k:i-1:-1]
    return new_route

# %%

def comp_shortest_path(T_start, T_end, cooling_factor, nMarkov, coords):
    """Function that computes the shortest path for the Traveling Salesman Problem using Simulated Annealing.


    Args:
        T_start (float): Starting temperature
        T_end (float): Temperature that determines when the Anealing process stops
        cooling_factor (float): determines the speed of the cooling process.
        nMarkov (int): Length of the markov chain
        coords (numpy.ndarray): coordinate array with tuples of city coordinates in the form: ()

    Returns:
        tuple: [0] final number of iterations, [1]minimal cost, [2]optimal cities list
    """
    T = T_start
    itr = 0
    matrix = distance_matrix(coords)

    route = np.arange(0,len(coords))
    np.random.shuffle(route)
    num_cities = len(route)
    # starts and end at the first node in route
    np.append(route,route[0])

    old_cost = total_dist(route,matrix)
    min_cost = old_cost

    best_cost_till_now = []
    best_for_Temp = []
    new_cost_arr = []
    new_cost = old_cost
    optimal_list_cities = None

    while T >= T_end:
        for k in range(nMarkov):    # Markov
            i,j = generate_i_j(num_cities)

            new_cost = old_cost - (matrix[route[i-1]][route[i]] + matrix[route[j+1]][route[j]]) + (matrix[route[i-1]][route[j]] + matrix[route[j+1]][route[i]])
            #print(new_cost)

            cost_difference = new_cost - old_cost

            new_cost_arr.append(new_cost)


            if(new_cost < min_cost):
            #print(min_cost)
            #min_cost_arr.append(new_cost)
                min_cost = new_cost


            prob = np.minimum(math.exp(-cost_difference/T),1)
            random_num = np.random.uniform()
            #print(prob)

            # Accept it
            if(random_num <= prob):
                route = two_opt_swap(route,i,j)
                old_cost = new_cost
                optimal_list_cities = route

        best_cost_till_now.append(min_cost)
        best_for_Temp.append(old_cost)

        itr = itr + 1
        T = T * cooling_factor
    return itr, min_cost, optimal_list_cities
# %%

# %%
