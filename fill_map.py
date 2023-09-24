# import open3d as o3d
# import random
# import numpy as np

# # x = []
# # z = []
# # num = 100
# # while num <= 140:
    # # num+=random.uniform(0.1, 0.3)
    # # x.append(num)

# # x = np.array(x)

# # x = np.linspace(100, 140, 200)
# # y = np.linspace(100, 101, 200)
# # z = np.linspace(2, 5, len(x))

# # np_3 = np.zeros((len(x), 1, 3), dtype=np.float16)

# # x = np.expand_dims(x, axis = 1)
# # y = np.expand_dims(y, axis = 1)
# # z = np.expand_dims(z, axis = 1)



# # np_3[:,:,0] = x
# # np_3[:,:,1] = y
# # np_3[:,:,2] = z

# # print(np_3.shape)

# # np_3 = np_3.squeeze(1)

# # print(np_3.shape)

# # print(np_3)
# # np.savetxt('qq.xyz',np_3, fmt='%5f')




# # pcd = o3d.io.read_point_cloud("qq.xyz")

# # o3d.visualization.draw_geometries([pcd])


# #-----------------------------------------------------------
# x = 100
# y = 120
# z = 3
# all_data = []

# for i in range(50):
    # x = 100
    # y+=random.uniform(0.3, 0.5)
    # z+=random.uniform(0.01, 0.03)
    # for j in range(50):
        # x+=random.uniform(0.5, 0.53)  
        # get_datax = list(range(100,150,1))
        # random.shuffle(get_datax)
        # get_datax = get_datax[:int((50)/2)]
        
        # get_datay = list(range(100,150,1))
        # random.shuffle(get_datax)
        # get_datay = get_datay[:int((50))]
        
        # if int(x) not in get_datax :
            # all_data.append([x, y+random.uniform(0.03, 0.08), z+random.uniform(0.01, 0.08)])



# all_data = np.array(all_data)
# np.savetxt('qq1.xyz',all_data, fmt='%5f')
# # pcd = o3d.io.read_point_cloud("qq.xyz")
# # o3d.visualization.draw_geometries([pcd])


# data = np.loadtxt('qq1.xyz')
# print(data.shape)

# sub_X = data[:,0][:-1]- data[:,0][1:]
# sub_Y = data[:,1][:-1]- data[:,0][1:]
# # print(data[:,2])

# print(maxdata[:,0])

# print(max(sub_X), min(sub_X))
# print(max(sub_Y), min(sub_Y))
# all_data = []
# temp_data = []
# for i in range(len(sub_X)):
    # # print('x', sub_X[i], 'y', sub_Y[i])
    # # print()
    # if sub_X[i]>0 and sub_Y[i]>0:
        # # np.savetxt(f'qq{i}_sub.xyz',temp_data, fmt='%5f')
        # # print(len(temp_data))
        # # print(temp_data, '\n')
        # all_data.append(temp_data)
        # temp_data=[]
    # else:
        # temp_data.append(list(data[i]))

# # 計算X之間的間隔
# def remove_Ouliers(gap):
    # #IQR = Q3-Q1
    # n=1.5
    # Q3 = np.percentile(gap,55)
    # Q1 = np.percentile(gap,45)
    # IQR = Q3 - Q1
    # print('IQR ', IQR)
    # #outlier = Q3 + n*IQR
    # print('Q3 ',  Q3 + n*IQR)
    # gap=gap[gap <  Q3 + n*IQR]
    # #outlier = Q1 - n*IQR 
    # print('Q1 ',Q1 - n*IQR )
    # gap=gap[gap > Q1 - n*IQR ]
    # # print ("Shape Of The After Ouliers: ",gap.shape)
    
    # return gap


# max_len = 0
# max_all_data=[]
# for i in range(len(all_data)):
    # if max_len<len(all_data):
        # max_all_data = all_data[i]
# max_all_data = np.array(max_all_data)

# max_all_data_X = max_all_data[:,0]
# max_all_data_Y = max_all_data[:,1]
# max_all_data_Z = max_all_data[:,2]
# gap_x = []
# for i in range(len(max_all_data_X)-1):
    # gap_x.append(max_all_data_X[i]-max_all_data_X[i+1])
# print('mid(gap_x) ', sorted(gap_x)[int(len(gap_x)/2)])
# remove_Ouliers_gap_x = remove_Ouliers(np.array(gap_x))
# print('mid(gap_x) ', sorted(remove_Ouliers_gap_x)[int(len(remove_Ouliers_gap_x)/2)])
# mid_gap_x = sorted(remove_Ouliers_gap_x)[int(len(remove_Ouliers_gap_x)/2)]

# print('------------- max_all_data -------\n',max_all_data)

# # 尋找大於gap的數值
# residual = 0.05
# find_index = np.where(gap_x/(mid_gap_x+residual)>1)[0]
# print(find_index)
# fill_data_X = []
# fill_data_Y = []
# fill_data_Z = []
# for index in find_index:
    # if index+1>=len(max_all_data_X):
        # pass
    # else:
        # print((max_all_data_X[index],max_all_data_X[index+1]), (max_all_data_X[index]-max_all_data_X[index+1]))
        # print(round((max_all_data_X[index]-max_all_data_X[index+1])/mid_gap_x-1))
        # print((max_all_data_X[index]-max_all_data_X[index+1])/mid_gap_x-1)
        # step = round((max_all_data_X[index]-max_all_data_X[index+1])/mid_gap_x-1)
        
        # fill_data_X.extend(list(np.linspace(max_all_data_X[index], max_all_data_X[index+1], step+2)[1:-1]))
        # fill_data_Y.extend(list(np.linspace(max_all_data_Y[index], max_all_data_Y[index+1], step+2)[1:-1]))
        # fill_data_Z.extend(list(np.linspace(7, 9, step+2)[1:-1]))

# fill_data_X = np.array(fill_data_X)
# fill_data_Y = np.array(fill_data_Y)
# fill_data_Z = np.array(fill_data_Z)

# print(fill_data_X)
# print(fill_data_Y)
# print(fill_data_Z)

# # 合體
# fill_data = np.zeros((len(fill_data_X),3), dtype=np.float32)
# fill_data[:,0] = fill_data_X
# fill_data[:,1] = fill_data_Y
# fill_data[:,2] = fill_data_Z
# print(fill_data)


# final_data = np.concatenate([max_all_data, fill_data])
# print(final_data)


# # 已經取得最大值的一列
# # all_data = np.array(all_data)
# for i in range(len(all_data)):
    # temp = np.array(all_data[i])
    # print('--------------all_data-------------\n', temp)


# pcd = o3d.io.read_point_cloud("qq1.xyz")
# o3d.visualization.draw_geometries([pcd])



import scipy.spatial as spt

def find_nearest(point, points, xyz):
    tree = spt.cKDTree(data=points)  

    distances, indexs = tree.query(point, k=2) 
    x = [point[0], points[indexs[1]][0]]
    y = [point[1], points[indexs[1]][1]]
    
    print('----------distances indexs---------')
    print(distances)
    print(indexs)
    print(point)
    for i in range(len(indexs)):
        print(points[indexs[i]])
        print(xyz[indexs[i]])
    


import numpy as np

data = np.loadtxt('qq1.xyz')
print(data.shape)

print(max(data[:,0]))
print(min(data[:,0]))
print(max(data[:,1]))
print(min(data[:,1]))

xy = []
for i in range(len(data)):
    xy.append([data[:,0][i],data[:,1][i]])
    
print(xy)
point = [max(data[:,0]), max(data[:,1])]
find_nearest(point, xy, data)