import random
import matplotlib.pyplot as plt
import numpy as np

def KDTree(points: list, nCuts: int, ending_boundary):
    """
    input: points: list of list that represents the coordinate of points, the dimension of the points determines the dimension of KD tree.
           nCuts: how many times the space will be divided. ex: depth = 1, there will be 1 cut, the space is divided into 2 parts. depth = 3, there will be 3 cuts, 
           the space is divided into 4 parts, etc.    
    output: two lists called boundary_start and boundary_end that corresponding to the starting points and ending points of the cuts 
    
    This algorithm gives the KD tree, output the boundary of space that represents the leaves of the KD tree. The stopping criteria is when it runs out of the cuts. 
    There might be the cases that a space only contains 1 point but the program wants to divde it, in this case the cut will be at the boundary, so the sapce will not be 
    further divided. 

    The function automatically detects the dimensions.
    
    one example: points = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
                nCuts = 3
    """
    # Output lists for the start and end boundaries of each block
    dim = len(points[0])

    current_start = [0 for _ in range(0, dim)]
    current_end = ending_boundary 

    areas = [(current_start,current_end)] # keep adding to this list and only return the last n+1
    
    for cut in range(0, nCuts):
        axis = int(int(np.log2(cut+1)) % dim)
        current_points = [point for point in points if all(current_start[i] <= point[i] < current_end[i] for i in range(dim))]

        current_points.sort(key=lambda x: x[axis])
        if len(current_points) == 0:
                cutting_point = current_start[axis]
        else:
            median_idx = len(current_points) // 2
            cutting_point = current_points[median_idx][axis]
        left_end = current_end.copy()
        left_end[axis] = cutting_point
        areas.append((current_start, left_end))

        right_start = current_start.copy()
        right_start[axis] = cutting_point
        areas.append((right_start, current_end))

        current_start, current_end = areas[areas.index((current_start,current_end)) + 1]

    return areas[-(nCuts + 1): ]




# ------------------------------------------------------------ Testing: if 2D case, draw graphs ------------------------------------------------------------
# num_points = 20  
# x_range = (0, 15)  
# y_range = (0, 15)  
# z_range = (0, 15)

# for _ in range(0, 100):
#     points = []
#     for _ in range(num_points):
#         x = random.randint(*x_range)
#         y = random.randint(*y_range)
#         z = random.randint(*z_range)
#         points.append([x, y, z])
#     nCut = random.randint(*(0, 10))
#     print('points:', points)
#     print('nCut:', nCut)

#     d = len(points[0])
#     result = KDTree(points, nCut, [max(point[i] for point in points) for i in range(0, d)])

#     if d == 2:
#         x_coords = [point[0] for point in points]
#         y_coords = [point[1] for point in points]
#         plt.figure(figsize=(max(x_coords) + 1, max(y_coords) + 1))
#         plt.scatter(x_coords, y_coords, color='blue', marker='o')

#     print("result:")
#     for area in result:
#         print(f"bottom left corner: {area[0]}, upper right corner: {area[1]}")        
#         if d == 2:  
#             rectangle = plt.Rectangle((area[0][0], area[0][1]), area[1][0] - area[0][0], area[1][1] - area[0][1], 
#                                 edgecolor='red', facecolor='none', linewidth=2)
#             plt.gca().add_patch(rectangle)  

#     if d == 2:
#         plt.grid(True)
#         plt.show()
