import gaussian_random_fields
import math
from classes import criticalPoint
import numpy as np
import gudhi
from sklearn.neighbors import KDTree
from collections import Counter
from unionfind import UnionFind
from collections import defaultdict
from copy import copy, deepcopy
import matplotlib.pyplot as plt


DECIMAL = 3
EPSILON = 10**(-DECIMAL)
FOUR_NBS = [[-1, 0], [0, -1], [0, 1], [1, 0]]



def find_critical_points_4_neighbors(gfield):
    """
    Input: 2D array, representing the matrix that we want to find critical points
    Output: a list of criticalPoint, identified by index in gfield.
    Find critical points (local min, local max, saddles) in the gfield, under the assumption that each point has 4 neighbors
    (up, down, left, right)
    """
    critical = []  # list of critical points, contains the index of position of critical points
    rows, cols = gfield.shape
    for i in range(0, rows):  # row
        for j in range(0, cols):  # column

            wedge = 0
            upper_star = []
            index = -1
            for k in [[i - 1, j], [i, j + 1], [i + 1, j], [i, j - 1]]:  # clockwise, starting from 12 o'clock
                try:
                    if k[0] < 0 or k[1] < 0 or k[0] >= rows or k[1] >= cols:
                        raise Exception()
                    if gfield[k[0]][k[1]] > gfield[i][j]:
                        upper_star.append(1)
                    else:
                        upper_star.append(-1)
                
                # if upper_star[-1] * upper_star[-2] < 0:
                #     wedge += 1
                    
                except:
                    continue
            
            for a in range(0, len(upper_star)):
                if upper_star[a] * upper_star[a-1] < 0:
                    wedge += 1
            
            if not int(math.ceil(wedge / 2)) == 1:
                if int(math.ceil(wedge / 2)) == 0:
                    if upper_star[0] > 0:
                        index = 0 # local min
                    else:
                        index = 2  # local max
                else:
                    index = 1  # saddle
                critical.append(criticalPoint(key=tuple((i,j)), value=gfield[i][j], wedge=int(math.ceil(wedge / 2)), index = index))
    return critical

def find_critical_points(gfield):
    """
    given the graph function from R^2 to R, find the critical points of gfield.
    critical points are identified by considering its 8 neighborhoods. Ie. saddle points alters.
    input: 2D array, gfield, represents the graph
    output: list of criticalPoints (class that stores points, key is the x,y index, value is the function value, the class also records the wedge and index)
    """
    critical = []  # list of critical points, contains the index of position of critical points
    upper_star = []
    rows, cols = gfield.shape    

    for i in range(0, rows):  # row
        for j in range(0, cols):  # column
            wedge = 0
            for k in [[i - 1, j], [i-1, j+1], [i, j + 1], [i + 1, j + 1], [i + 1, j], [i+1, j-1], [i, j - 1], [i - 1, j - 1]]:  # clockwise, starting from 12 o'clock
                try:
                    if k[0] < 0 or k[1] < 0 or k[0] >= rows or k[1] >= cols:
                        raise Exception()
                    if gfield[k[0]][k[1]] > gfield[i][j]:
                        upper_star.append(1)
                    else:
                        upper_star.append(-1)
                    
                    if upper_star[-1] * upper_star[-2] < 0:
                        wedge += 1
                        
                except:
                    continue
            if not int(math.ceil(wedge / 2)) == 1:
                critical.append(criticalPoint(key=tuple((i,j)), value=gfield[i][j], wedge=int(math.ceil(wedge / 2))))
    return critical

def find_gradient_vector_field(gfield):
    """
    Given the graph of a function, output its gradient vector field, by definition of gradient. Ie. limit f(x+h) - f(x) / ||h||, but use discrete way
    input: 2D array, gfield, represents the graph
    output: 2 2D array, represents the x-coordinate and y-coordinate of the gradient function: R^2 -> R^2. 
    """
    rows, cols = gfield.shape    
    grad_x = np.zeros((rows, cols))
    grad_y = np.zeros((rows, cols))
    
    shift_up_gfield = np.roll(gfield, -1, axis=0)
    shift_up_gfield[-1] = gfield[-1]

    shift_left_gfield = np.roll(gfield, -1, axis=1)
    shift_left_gfield[:, -1] = gfield[:, -1]

    grad_x = shift_up_gfield - gfield
    grad_y = shift_left_gfield - gfield
    return grad_x, grad_y

def create_component(uf, gfield, a):
    """
    uf: union find structure
    gfield: the 2D array
    a: stopping criteria 
    """
    for i in range(1, len(gfield)-1): # row
        for j in range(1, len(gfield[i])-1): # col
            if gfield[i][j] <= a:
                for k,l in [[-1,-1], [-1, 0], [-1, 1],
                            [0, -1],          [0, 1],
                            [1, -1], [1, 0],  [1, 1]]: 
                    if gfield[i+k][j+l] <= a and (i+k)*len(gfield) + j+l in uf:
                        uf.union(i*len(gfield) + j, (i+k)*len(gfield) + j + l)
    return uf

def find_largest_component(uf:UnionFind, glength, c_uf_index, point_index, f_0_):
    """
    Create component for each level, and if the index of the component is 0, then return this component and its level. 
    """
    # total_components = {}
    glength = list(set(glength))
    glength.sort(reverse=True)
    for l in glength:
        uf = create_component(uf, f_0_, l)
        for c in c_uf_index:
            s = sum([point_index[i] for i in range(0, len(c_uf_index)) if uf.connected(c, c_uf_index[i])])
            if s == 0:
                return uf.component(c), l
                
    return set(), -1

def are_points_on_same_side(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    let (x3, y3), (x4, y4) be a line segment, return if (x1, y1) and (x2, y2) are on the same side of this line segment
    """
    # Compute cross product for two vectors (x4-x3, y4-y3) and (x1-x3, y1-y3), see the orientation
    det1 = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
    # Compute cross product for two vectors (x4-x3, y4-y3) and (x2-x3, y2-y3), see the orientation
    det2 = (x4 - x3) * (y2 - y3) - (y4 - y3) * (x2 - x3)
    # on the same side if they have same orientation, different if they have different orientation
    return det1 * det2 > 0

def merge_tree_archived(uf, gfield):
    """
    return a list of birth, death of the graph, represents the 0-persistence
    """
    persistence = []
    non_closed_per = {}
    a_list = list(set(gfield.flatten()))
    a_list.sort()
    a_list.append(a_list[-1] +1)
    current_component = []
    for a in range(1, len(a_list)):
        for i in range(0, len(gfield)): # row
            for j in range(0, len(gfield[i])): # col
                if a_list[a-1] <= gfield[i][j] < a_list[a]:
                    for k,l in [[-1,-1], [-1, 0], [-1, 1],
                                [0, -1],          [0, 1],
                                [1, -1], [1, 0],  [1, 1]]: 
                        if 0 <= i+k < len(gfield) and 0 <= j+l < len(gfield[0]):
                            if gfield[i+k][j+l] < a_list[a]:
                                uf.union(i*len(gfield[0]) + j, (i+k)*len(gfield[0]) + j + l)
        # check if the structure of the tree change 
        current_component = [
            tuple(val) for k, val in uf.component_mapping().items() if gfield[k // len(gfield[0])][k % len(gfield[0])] < a_list[a]  # Filter by key
           # for x in tuple(sorted(val) for val in [val])   # Remove duplicates
        ] 
        current_component = list(set(current_component))
   #     print('current component', current_component)
    #    print('non_closed_per', non_closed_per)      
  
        for k in current_component:
            subset = [key for key in non_closed_per.keys() if set(key) < set(k)]
            if len(subset) > 1:
                filtered_dict = {k: v for k, v in non_closed_per.items() if k in subset}
                # persistence.append([non_closed_per[max(filtered_dict, key=filtered_dict.get)], a_list[a]])
                min_to_keep = min(filtered_dict, key=filtered_dict.get)
                non_closed_per[k] =  non_closed_per.pop(min_to_keep)
                subset.remove(min_to_keep)
                for s in subset:
                    persistence.append([non_closed_per.pop(s), a_list[a-1]])
                
            elif len(subset) == 1:
                non_closed_per[k] = non_closed_per[subset[0]]
                non_closed_per.pop(subset[0])
            else:
                if k not in non_closed_per.keys():
                    non_closed_per[k] = a_list[a-1]
            
    for key in non_closed_per.keys():
        persistence.append([non_closed_per[key], a_list[-1]])
        # non_closed_per.pop(key)
    return persistence

def merge_tree(uf, gfield, four_neighbors):
    """
    return a list of birth, death of the graph, represents the 0-persistence
    """
    rows, cols = gfield.shape
    persistence = []
    persistence_index = []
    non_closed_per = {}
    a_list = list(set(gfield.flatten()))
    a_list.sort()
    if four_neighbors:
        neighbors = FOUR_NBS
    else:
        neighbors = [[-1,-1], [-1, 0], [-1, 1],
                     [0, -1],          [0, 1],
                     [1, -1], [1, 0],  [1, 1]]
    for a in a_list:
        indices = np.where(gfield == a)
        index_list = list(zip(indices[0], indices[1]))
        for i,j in index_list:
            for k,l in neighbors:
                if 0<= i+k < rows and 0 <= j+l < cols and gfield[i+k][j+l] <= a:
                     uf.union((i+k)*cols + j+l, (i)*cols + j)
            k = tuple(uf.component(i*cols + j))
            subset = [key for key in non_closed_per.keys() if set(key) < set(k)]
            if len(subset) > 1:
                filtered_dict = {k: v for k, v in non_closed_per.items() if k in subset}
                min_key = None
                min_value = float('inf')
                for key, index in filtered_dict.items():
                    value = gfield[index[0]][index[1]]  # Get the value from gfield at the 2D index
                    if value < min_value:
                        min_value = value
                        min_key = key
                non_closed_per[k] =  non_closed_per.pop(min_key)
                subset.remove(min_key)
                for s in subset:
                    m, n = non_closed_per.pop(s)
                    persistence_index.append([(m,n), (i,j)])
                    persistence.append([gfield[m][n], a])
                
            elif len(subset) == 1:
                non_closed_per[k] = non_closed_per.pop(subset[0])
                
            else:
                if k not in non_closed_per.keys():
                    non_closed_per[k] = (i,j)
            
    for key in non_closed_per.keys():
        persistence_index.append([non_closed_per[key], math.inf])
        persistence.append([gfield[non_closed_per[key][0]][non_closed_per[key][1]], math.inf])
    return persistence_index, persistence

def find_path_archive(gfield: np.array, low_index, high_index):
    """p1, p2 represent the index of two points in the gfield""" 
    path = []
    rows, cols = gfield.shape
    cur_index = low_index
    while cur_index != high_index:
        path.append(cur_index)
        possible_four_nbs = [(k,l) for k,l in FOUR_NBS if 0<= cur_index[0]+k < rows and 0 <= cur_index[1]+l <cols]  # in case on the boundary
        nbs_value = [gfield[cur_index[0]+k][cur_index[1]+l] for k,l in possible_four_nbs]
        next_value = min([gfield[cur_index[0]+k][cur_index[1]+l] for k,l in possible_four_nbs if (cur_index[0]+k, cur_index[1]+l) not in path])
        next_index = possible_four_nbs[nbs_value.index(next_value)]
        cur_index = (cur_index[0] + next_index[0], cur_index[1] + next_index[1])
        if next_value == gfield[high_index[0]][high_index[1]] and cur_index != high_index:
            # TODO: this code has not been tested
            possible_next_step = [(cur_index[0]+k, cur_index[1]+l) for k,l in possible_four_nbs if (cur_index[0]+k, cur_index[1]+l) not in path and gfield[cur_index[0]+k][cur_index[1]+l] == next_value]
            min_step = abs(possible_next_step[0][0]-high_index[0]) + abs(possible_next_step[0][1]-high_index[1])
            next_index = possible_next_step[0]
            for m,n in possible_next_step:
                step = abs(m-high_index[0]) + abs(n-high_index[1])
                if step < min_step:
                    min_step = step
                    next_index = (m,n)
            cur_index = (cur_index[0] + next_index[0], cur_index[1] + next_index[1])

           
    path.append(high_index)
    return path

def find_path(gfield: np.array, low_index: tuple, high_index: tuple, debugmode:bool = False):
    """ 
    Input: gfield - the 2D array representing the graph of a function.
            low_index - the index of the point with lower value in the persistence pair in gfield.
            high_index - the index of the point with higher value in the persistence pair in gfield.
            debugmode - if print the intermediate steps of finding the path.
    Output: a list of indicies (tuple), representing the path. Each two consecutive element in the list are 
    neighbors each other.
    find the shortest path that flows from low_index to high_index, representing the arc in morse-smale complex.
    Using BFS:finding the shortest path that connects low_index and high_index such that each step in this path
    has value smaller than the value of high_index.
    """
    m, n = gfield.shape
    grid = (gfield <= gfield[high_index]).astype(int)  # 1=TRUE, low_index -> high_index
    if debugmode: print(grid)
    visited = set()
    queue = [(low_index, [low_index])]
    while queue:
        (x, y), path = queue.pop(0)
        visited.add((x, y))
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            x_, y_ = x + dx, y + dy
            if (x_, y_) == high_index:  # early exit
                return path + [(x_, y_)]
            if 0 <= x_ < m and 0 <= y_ < n and grid[(x_, y_)] == 1 and (x_, y_) not in visited:
                queue.append(((x_, y_), path + [(x_, y_)]))

    return []  # not found  

def find_type_of_one_point(p_value, neighbors_value):
    """
    find the type of a critical point.
    
    test cases:
    print(find_type_of_one_point(1, [0,0,0,0]))
    print(find_type_of_one_point(1, [1,2,3,4]))
    print(find_type_of_one_point(1, [2,2,2,0]))
    print(find_type_of_one_point(1, [2,2,0,0]))
    print(find_type_of_one_point(1, [2,0,2,0]))
    print(find_type_of_one_point(1, [0,2,0,0]))
    """
    wedge = 0
    upper_star = []
    for k in neighbors_value:  # clockwise, starting from 12 o'clock
        if k > p_value:
            upper_star.append(1)
        else:
            upper_star.append(-1)
    
    for a in range(0, len(upper_star)):
        if upper_star[a] * upper_star[a-1] < 0:
            wedge += 1
    
    if not int(math.ceil(wedge / 2)) == 1:
        if int(math.ceil(wedge / 2)) == 0:
            if upper_star[0] > 0:
                return 'local min'
            else:
                return 'local max'
        else:
            return 'saddle'
    else:
        return 'regular'

def find_range(gfield, p_index, nbs_p):
    '''need one global parameter EPSILON'''
    neighbors = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # clockwise
    lp_list = [-math.inf]
    hp_list = [math.inf]
    rows, cols = gfield.shape
    for i, n in enumerate(nbs_p):
        if gfield[n[0]+p_index[0]][n[1]+p_index[1]] > gfield[p_index[0]][p_index[1]]:
            # range_p = [-math.inf, gfield[n[0]+p_index[0]][n[1]+p_index[1]] - EPSILON]
            lp_list.append(-math.inf)
            hp_list.append(gfield[n[0]+p_index[0]][n[1]+p_index[1]] - EPSILON)
        else:
            # range_p = [gfield[n[0]+p_index[0]][n[1]+p_index[1]], math.inf]
            lp_list.append(gfield[n[0]+p_index[0]][n[1]+p_index[1]])
            hp_list.append(math.inf)
        neighbors = neighbors[neighbors.index((n[0], n[1])):] + neighbors[0: neighbors.index((n[0], n[1]))] # reorder the 4 neighbor such that this neighbor n is the first element in the list
        neighbors = [(-k[0], -k[1]) for k in neighbors]  # multiply the index by -1, change the neighbors to be the neighbor of n
        # neighbors = neighbors[neighbors.index([n[0], n[1]]):] + neighbors[0: neighbors.index([n[0], n[1]])]  # rorate such that this n is the first element
        neighbors_values = [gfield[n[0]+p_index[0] + k[0]][n[1]+p_index[1]+k[1]] for k in neighbors if 0<= n[0]+p_index[0] + k[0] <rows and 0<= n[1]+p_index[1]+k[1] < cols]
        neighbors_values2 = neighbors_values.copy()
        neighbors_values2[0] = gfield[n[0]+p_index[0]][n[1]+p_index[1]] + 1
        if find_type_of_one_point(gfield[n[0]+p_index[0]][n[1]+p_index[1]], neighbors_values) == find_type_of_one_point(gfield[n[0]+p_index[0]][n[1]+p_index[1]], neighbors_values2):
            # range_p = [-math.inf, math.inf]
            lp_list[i] = -math.inf
            hp_list[i] = math.inf
    return max(lp_list), min(hp_list)


def pair_cancellation(original_field, p1, p2):
    """
    Input: 2D arrary representing the graph of a function
    p1, p2: two indicies in the original_field that forms a persistence
    """
    gfield = deepcopy(original_field)
    rows, cols = gfield.shape
    if gfield[p1] > gfield[p2]:
        low = p2
        high = p1
    else:
        low = p1
        high = p2
    path_list = find_path(gfield, low, high)
    # print(path_list)
    possible_nbs_low = [(k,l) for k,l in FOUR_NBS if 0<= low[0]+k < rows and 0 <= low[1]+l <cols]  # in case on the boundary
    possible_nbs_high = [(k,l) for k,l in FOUR_NBS if 0<= high[0]+k < rows and 0 <= high[1]+l <cols]  # in case on the boundary
    nbs_low = [(low[0]+k, low[1]+l) for k, l in possible_nbs_low if (low[0]+k, low[1]+l) not in path_list]  # neighbors that are not in path list
    nbs_high = [(high[0]+k, high[1]+l) for k, l in possible_nbs_high if (high[0]+k, high[1]+l) not in path_list]
    min_nbs_low = min([gfield[a][b] for a,b in nbs_low])
    min_nbs_high = min([gfield[a][b] for a,b in nbs_high])
    upperbound = min_nbs_low
    if min_nbs_high > min_nbs_low:
        print('cannot cancelling the pair')
        return original_field
    for i in range(0, len(path_list)):
        avg_change = round((upperbound-min_nbs_high) / (len(path_list)-i+1), DECIMAL)
        p = path_list[i]
        nbs_p = [(k, l) for k, l in FOUR_NBS if (p[0]+k, p[1]+l) not in path_list and 0<= p[0]+k < rows and 0<= p[1]+l < cols]
        lp, hp = find_range(gfield, p,nbs_p)
        lp = max(lp, min_nbs_high)
        if i == 0:
            hp = min(hp, min_nbs_low-EPSILON)
        else:
            hp = min(hp, upperbound)
        if  lp <= hp:
            if i == 0:
                gfield[p[0]][p[1]] = hp
            else:
                if lp <= (upperbound - avg_change) <= hp:
                    gfield[p] = upperbound - avg_change
                    # gfield = np.round(gfield, DECIMAL)
                elif (upperbound - avg_change) < lp:
                    gfield[p] = lp
                else:
                    gfield[p] = hp
            upperbound = round(gfield[p], DECIMAL)
        else:
            print('cannot cancelling the pair')
            return original_field
    return gfield






def PH(field, four_neighbors: bool = True):
    """
    compute the Morse-Smale PH of a function: R^2 -> R
    """
    indices = [i for i in range(0, field.size)]
    uf_front = UnionFind(indices)
    p0_index, p0 = merge_tree(uf_front, gfield=field, four_neighbors = four_neighbors)
    uf_back = UnionFind(indices)
    neg_p0_index, neg_p0 = merge_tree(uf_back, gfield=-field, four_neighbors = four_neighbors)
    p1 = [[-p[1], -p[0]] for p in neg_p0 if not math.inf in p and not -math.inf in p]
    return p0, p1


# ---------------------------------main---------------------------------
# gfield = np.array([[-0.44, -0.15,  0.22,  0.79,  0.79,  0.66,  0.62, 0.72,  0.15,
#         -0.12,  0.42,  1.35],
#        [-0.76, -1.03, -0.32, -0.24, -0.38, -0.16,  0.66,  0.59,  0.33,
#          0.73, -0.06,  0.3 ],
#        [-0.98, -1.5 , -1.31, -1.17, -1.09, -0.3 ,  0.69,  1.23,  0.73,
#          0.12, -0.19, -0.4 ],
#        [-1.23, -1.49, -1.51, -2.01, -1.56, -0.31,  0.02,  0.43,  0.26,
#         -0.24,  0.26,  0.2 ],
#        [-1.04, -1.49, -1.63, -1.86, -1.34, -0.99,  0.14,  1.07,  0.3 ,
#          0.19,  0.53,  1.36],
#        [-1.64, -1.29, -1.41, -1.79, -1.63, -0.87, -0.02,  0.76,  0.56,
#          0.3 ,  0.38,  0.85],
#        [-2.25, -1.25, -1.3 , -2.02, -0.84, -0.7 ,  0.3 ,  0.74,  1.13,
#          0.34,  0.44,  0.62],
#        [-2.17, -1.44, -2.25, -2.28, -1.31, -0.77, -0.6 ,  0.26,  0.22,
#          0.46,  0.57,  1.3 ],
#        [-1.99, -2.1 , -1.73, -1.51, -1.15, -1.09, -0.73, -0.15,  0.35,
#         -0.1 ,  0.12,  0.37],
#        [-1.58, -1.18, -1.01, -1.15, -0.7 , -0.81, -0.03,  0.14, -0.11,
#         -0.24,  0.09,  0.36]])


SIZE= 20
np.random.seed(42)
gfield = gaussian_random_fields.gaussian_random_field(size = SIZE)
gfield = np.round(gfield, 2)
rows, cols = gfield.shape

# identify all the critical points
critical_points = find_critical_points_4_neighbors(gfield=gfield)
critical_dic = {}  # store the information of critical point, key is the xyz axis
for c in critical_points:
    critical_dic[(c.key[0], c.key[1], c.value)] = c

# construct 2D KD tree
LeafSize = 15 # Number of points at which to switch to brute-force 
c_position = [[c[0], c[1]] for c in critical_dic.keys()]
twoDTree = KDTree(np.array(c_position), leaf_size= LeafSize)
twoDTree.get_arrays()[2]
leaf = [s[2] for s in twoDTree.get_arrays()[2]]  # only need leaf nodes
bound_start, bound_end = twoDTree.get_arrays()[3][0], twoDTree.get_arrays()[3][1]
bound_start_leaf = [list(bound_start[i]) for i in range(0, len(leaf)) if leaf[i] == 1]
bound_end_leaf = [list(bound_end[i]) for i in range(0, len(leaf)) if leaf[i] == 1]
matrix_list = []
for i in range(0, len(bound_start_leaf)):
    r_s, c_s = bound_start_leaf[i]
    r_e, c_e = bound_end_leaf[i]
    matrix_list.append(gfield[int(r_s): int(r_e), int(c_s): int(c_e)])


new_gfield_list = []

p0, p1 = PH(gfield, True)


for j, field in enumerate(matrix_list):
    indices_ = [i for i in range(0, field.size)]
    f1 = UnionFind(indices_)
    indices, ph = merge_tree(f1, field, four_neighbors=True)
    values = [p[1]-p[0] for p in ph]
    sorted_indices = [index for index, value in sorted(zip(indices, values), key=lambda x: x[1])]
    for i, pair in enumerate(sorted_indices):
        # if i == 6:
        #     print('pause')
        if math.inf in pair or -math.inf in pair:
            print('PH0: for gfield' + str(j), 'pair', i, 'out of' , len(sorted_indices), 'done')
            continue
        field = pair_cancellation(field, pair[0], pair[1]) 
        # print(field, pair)
        print('PH0: for gfield' + str(j), 'pair', i, 'out of' , len(sorted_indices), 'done')
        # print(gfield[pair[0][0]][pair[0][1]], gfield[pair[1][0]][pair[1][1]])
    # print(np.round(field, 2))
    f2 = UnionFind(indices_)
    neg_field = -field
    indices, ph = merge_tree(f2, neg_field, four_neighbors=True)
    values = [p[1]-p[0] for p in ph]
    sorted_indices = [index for index, value in sorted(zip(indices, values), key=lambda x: x[1])]
    for i, pair in enumerate(sorted_indices):
        if math.inf in pair or -math.inf in pair:
            print('PH1: for gfield' + str(j), 'pair', i, 'out of' , len(sorted_indices), 'done')
            continue
        neg_field = pair_cancellation(neg_field, pair[0], pair[1]) 
        
        print('PH1: for gfield' + str(j), 'pair', i, 'out of' , len(sorted_indices), 'done')
    new_gfield_list.append(-neg_field)

# concatenate together
new_gfield = np.zeros((rows, cols))
for i in range(0, len(bound_start_leaf)):
    row_start, col_start = bound_start_leaf[i]
    row_end, col_end = bound_end_leaf[i]
    new_gfield[int(row_start):int(row_end), int(col_start):int(col_end)] = new_gfield_list[i]

# # concatenated13 = np.concatenate((new_gfield_list[0], new_gfield_list[2]), axis=0)
# # concatenated24 = np.concatenate((new_gfield_list[1], new_gfield_list[3]), axis=0)
# # new_gfield = np.concatenate((concatenated13, concatenated24), axis=1)
# # new_gfield = np.round(new_gfield, 4)
# # print(new_gfield)
# # difference = np.abs(new_gfield - gfield)
# # max_difference = np.max(difference)
# # print('the max difference is', max_difference)


new_p0, new_p1 = PH(new_gfield, True)
bottleneck = gudhi.bottleneck_distance(p0+p1, new_p0+new_p1)
print('the bottleneck distance is', bottleneck)

# calculate true pair cancellation 
gfield_can = deepcopy(gfield)
indices_ = [i for i in range(0, gfield_can.size)]
f1 = UnionFind(indices_)
indices, ph = merge_tree(f1, gfield_can, four_neighbors=True)
values = [p[1]-p[0] for p in ph]
sorted_indices = [index for index, value in sorted(zip(indices, values), key=lambda x: x[1])]
for i, pair in enumerate(sorted_indices):
    if math.inf in pair or -math.inf in pair:
        print('PH0: for gfield_can', 'pair', i, 'out of' , len(sorted_indices), 'done')
        continue
    gfield_can = pair_cancellation(gfield_can, pair[0], pair[1]) 
    print('PH0: for gfield_can', 'pair', i, 'out of' , len(sorted_indices), 'done')
f2 = UnionFind(indices_)
neg_field = -gfield_can
indices, ph = merge_tree(f2, neg_field, four_neighbors=True)
values = [p[1]-p[0] for p in ph]
sorted_indices = [index for index, value in sorted(zip(indices, values), key=lambda x: x[1])]
for i, pair in enumerate(sorted_indices):
    if math.inf in pair or -math.inf in pair:
        print('PH1: for gfield_can', 'pair', i, 'out of' , len(sorted_indices), 'done')
        continue
    neg_field = pair_cancellation(neg_field, pair[0], pair[1]) 
    print('PH1: for gfield_can', 'pair', i, 'out of' , len(sorted_indices), 'done')
gfield_can = -neg_field


figure, axis = plt.subplots(1, 3)

axis[0].imshow(gfield)
axis[0].set_title('Heatmap for Input Matrix')
axis[1].imshow(new_gfield)
axis[1].set_title('Heatmap for Output Matrix')
axis[2].imshow(gfield_can)
axis[2].set_title('Heatmap for Matrix with Overall Cancellation')
plt.show()













