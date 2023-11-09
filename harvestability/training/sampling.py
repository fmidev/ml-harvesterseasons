from geopy.distance import distance
import pandas as pd
import numpy as np
import os
import math

train_data = "/home/ubuntu/data/ml-harvestability/training/soildata/train_locations.txt"

lucas_points = "/home/ubuntu/data/ml-harvestability/training/soildata/LUCAS_train.csv"

ground_truth_locations = "/home/ubuntu/data/ml-harvestability/training/soildata/harvestability_gt_locations.txt"

cols_own = ["TH_LONG","TH_LAT"]

cols_use = ["TH_LONG","TH_LAT","harvestability"]

lucas_df = pd.read_csv(lucas_points,usecols=cols_use)

lucas_df = lucas_df.loc[~lucas_df["harvestability"].isin([254,0])]


lucas_coords = lucas_df[cols_own].values.tolist()


def findpoints(lon, lat):
    radius = 1
    N = 500 

    # generate points
    circlePoints = []
    for k in range(N):
        angle = math.pi*2*k/N
        dx = radius*math.cos(angle)
        dy = radius*math.sin(angle)
        y = lat + (180/math.pi)*(dy/6371) #Earth Radius
        x = lon + (180/math.pi)*(dx/6371)/math.cos(lon*math.pi/180) #Earth Radius
        # add to list
        circlePoints.append([x,y])

    return circlePoints

# reading ground truth as chunk by chunk because of the volume
# chunksize = 10 ** 7
# points_filtered = []
# points_values = []

# for each_loc in lucas_coords:
#     count = 0
#     found = False
#     while count<=200:
#         for chunk in pd.read_csv(ground_truth_locations,header=None,
#                                 sep="\s+|,",
#                                 names=cols_use,
#                                 chunksize=chunksize):
#             chunk_coords = chunk[["TH_LONG","TH_LAT"]].values.tolist()
#             if each_loc in chunk_coords:
#                 chunk_values = chunk[["harvestability"]].values.tolist()
#                 for point in chunk_coords:
#                     if distance(each_loc, point).km < 1:
#                         points_filtered.append(point)
#                         index_point =chunk_coords.index(point)
#                         points_values.append(chunk_values[index_point])
#                         count+=1
#     print(len(points_filtered))
#     print(points_filtered[:5])
#     break

points_filtered = []

for each_loc in lucas_coords:
    points_filtered.extend(findpoints(each_loc[0], each_loc[1]))

train_file = open(train_data,"w")

if len(points_filtered)>0:

    for i in range(len(points_filtered)):

        train_file.write(str(points_filtered[i][0])+' '+str(points_filtered[i][1]))
        train_file.write("\n")

train_file.close()

        
    




