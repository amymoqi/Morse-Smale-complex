
import gaussian_random_fields
import math
import numpy as np
import gudhi
from unionfind import UnionFind
from copy import deepcopy
import matplotlib.pyplot as plt
from implementation import *
from PIL import Image
import time


DECIMAL = 3
EPSILON = 10**(-DECIMAL)
FOUR_NBS = [[-1, 0], [0, -1], [0, 1], [1, 0]]

def image_input(location: str, rows, cols):
    img = Image.open(location)
    numpydata = np.asarray(img)
    red = []
    green = []
    blue = []
    for i in range(0,len(numpydata)):
        red.append([])
        green.append([])
        blue.append([])
        for j in range(0, len(numpydata[i])):
            red[i].append(numpydata[i][j][0])
            green[i].append(numpydata[i][j][1])
            blue[i].append(numpydata[i][j][2])
    return np.array(red)[:rows, :cols], np.array(green)[:rows, :cols], np.array(blue)[:rows, :cols]


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


SIZE= 50
np.random.seed(42)
gfield = gaussian_random_fields.gaussian_random_field(size = SIZE)
gfield = np.round(gfield, 2)

# gfield, green, blue  = image_input('Image_20230907141610.jpg', SIZE, SIZE)

rows, cols = gfield.shape

start_time = time.time()

# identify all the critical points
critical_points = find_critical_points_4_neighbors(gfield=gfield)
critical_dic = {}  # store the information of critical point, key is the xyz axis
for c in critical_points:
    critical_dic[(c.key[0], c.key[1], c.value)] = c

# construct 2D KD tree
nCut = 3 # how many times the space will be cutted, ie: 3 cuts will produce 4 blocks. 
c_position = [[c[0], c[1]] for c in critical_dic.keys()]
blocks = KDTree(points = c_position, nCuts = nCut, ending_boundary= [rows, cols])
# blocks = KDTree(points = c_position, nCuts = nCut, ending_boundary= [rows, cols, gfield.max])  # 3D cases

matrix_list = []
for block in blocks:
    r_s, c_s = block[0][0], block[0][1]
    r_e, c_e = block[1][0], block[1][1]
    matrix_list.append(gfield[r_s: r_e, c_s: c_e])

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
for i, block in enumerate(blocks):
    r_s, c_s = block[0][0], block[0][1]
    r_e, c_e = block[1][0], block[1][1]
    new_gfield[r_s: r_e, c_s:c_e] = new_gfield_list[i]

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

end_time = time.time()


figure, axis = plt.subplots(1, 3)

axis[0].imshow(gfield)
axis[0].set_title('Heatmap for Input Matrix')
axis[1].imshow(new_gfield)
axis[1].set_title('Heatmap for Output Matrix')
axis[2].imshow(gfield_can)
axis[2].set_title('Heatmap for Matrix with Overall Cancellation')
plt.show()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.6f} seconds")