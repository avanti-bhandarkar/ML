
#Implementing a K-Nearest Neighbour classifier from scratch

from math import sqrt
import pandas as pd

df = pd.read_csv('test_knn.csv')
df.head()

def euclidean_distance(row1,row2):
  dist = 0.0
  length = len(row1)
  for i in range(0,length-1):
    temp1=(row1[i] - row2[i])**2
    dist = dist + temp1
    euc_dist = sqrt(dist)
  return euc_dist

test_data=df.to_numpy()
row0 = test_data[0]
for row in test_data:
  distance = euclidean_distance(row0,row)
  print(distance)

"""To understand how the above code works"""

distance = list()
dist = euclidean_distance(test_data[0],test_data[1])
distance.append((test_data[1],dist))
distance

distance = list()
dist = euclidean_distance(test_data[0],test_data[2])
distance.append((test_data[2],dist))
distance

distance = list()
dist = euclidean_distance(test_data[0],test_data[3])
distance.append((test_data[3],dist))
distance

def get_neighbours(train,test_row,num_neighbours):
  distances = list() # initialize a list
  for train_row in train:
    dist = euclidean_distance(test_row,train_row) # to find euclidean distance of each element from the test sample
    distances.append((train_row,dist))
  # to sort the distances
  distances.sort(key = lambda tup:tup[1])
  neighbours = list()
  for i in range(num_neighbours):
    neighbours.append(distances[i][0])
  return neighbours,distances

distance.sort(key = lambda tup:tup[1])
print('\n', distance)
neighbours,ecl_dist = get_neighbours(test_data,test_data[0],3)

print('\n Distances \n')
for d in ecl_dist:
  print (d)


print('\n K-nearest neighbours \n')
for n in neighbours:
  print (n)

def predict_classification(train,test_row,num_neighbors):
    neighbours,ecl_dist = get_neighbours(train,test_row,num_neighbors)
    output_values = [row[-1] for row in neighbours]
    prediction = max(set(output_values), key = output_values.count)
    return prediction

prediction = predict_classification(test_data, test_data[8],3)
print('Expected value: %d, predicted value: %d.' %(test_data[8][-1],prediction))
