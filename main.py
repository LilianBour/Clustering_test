import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans
dataset = pd.read_csv("iris_csv.csv")
#STATISTICS
#Head
print("Head : \n",dataset.head())

#Check null values
print("Null values : \n",dataset.isnull().sum())

#Descriptive statistic
print("\nStatistics :\n",dataset.describe(include='all'))

#dataset.plot(kind="hist")
plt.show()

#String -> int
print(dataset.iloc[:,4].unique())
set={'Iris-setosa':1,'Iris-versicolor':2,'Iris-virginica':3}
dataset=dataset.replace({'class': set})

x=dataset.iloc[:,0:4]
y=dataset.iloc[:,4]

#Scatterplot for each feature
"""plt.figure(1)
plt.scatter(x.iloc[:,0], x.iloc[:,1], c=y, cmap='jet')
plt.xlabel('sepallength', fontsize=11)
plt.ylabel('sepalwidth', fontsize=11)

plt.figure(2)
plt.scatter(x.iloc[:,2], x.iloc[:,3], c=y, cmap='jet')
plt.xlabel('petallength', fontsize=11)
plt.ylabel('petalwidth', fontsize=11)

plt.figure(3)
plt.scatter(x.iloc[:,0], x.iloc[:,2], c=y, cmap='jet')
plt.xlabel('sepallength', fontsize=11)
plt.ylabel('petallength', fontsize=11)

plt.figure(4)
plt.scatter(x.iloc[:,1], x.iloc[:,3], c=y, cmap='jet')
plt.xlabel('sepalwidth', fontsize=11)
plt.ylabel('petalwidth', fontsize=11)

plt.figure(5)
plt.scatter(x.iloc[:,0], x.iloc[:,3], c=y, cmap='jet')
plt.xlabel('sepallength', fontsize=11)
plt.ylabel('petalwidth', fontsize=11)

plt.figure(6)
plt.scatter(x.iloc[:,1], x.iloc[:,2], c=y, cmap='jet')
plt.xlabel('sepalwidth', fontsize=11)
plt.ylabel('petallength', fontsize=11)
plt.show()"""


#KMEANS
#finding the best k
cost=[]
for i in range(1,11):
    km = KMeans(n_clusters = i, n_jobs = 4, random_state=21)
    km.fit(x)
    #calculates squared error for the clustered points
    cost.append(km.inertia_)
# plot the cost against K values
plt.plot(range(1, 11), cost, color ='b', linewidth ='4')
plt.xlabel("Value of K")
plt.ylabel("Sqaured Error (=Cost)")
plt.show()

#Auto select best k
print(cost)
cost_in_perc=[]
k=0
counter=0
for i in cost :
    #Pass first value
    if counter !=0:
        #current cost at i in percentage
        c_cost = 1-(cost[counter]/cost[counter-1])
        # if not empty and if gain < 15% take k = i-1 <- PERCENTAGE HERE
        if len(cost_in_perc) !=0  and cost_in_perc[-1]- c_cost< 0.15 :
            k=counter-1
            break
        cost_in_perc.append(c_cost)
    counter=counter+1

print("Optimal k :",k)
km = KMeans(n_clusters=k, n_jobs=4, random_state=21)
km.fit(x)
centers = km.cluster_centers_
print(centers)

new_labels = km.labels_
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].scatter(x.iloc[:, 0], x.iloc[:, 1], c=y, cmap='gist_rainbow', edgecolor='k', s=150)
axes[1].scatter(x.iloc[:, 0], x.iloc[:, 1], c=new_labels, cmap='jet', edgecolor='k', s=150)
axes[0].set_xlabel('Sepal length', fontsize=18)
axes[0].set_ylabel('Sepal width', fontsize=18)
axes[1].set_xlabel('Sepal length', fontsize=18)
axes[1].set_ylabel('Sepal width', fontsize=18)
axes[0].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
axes[1].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
axes[0].set_title('Actual', fontsize=18)
axes[1].set_title('Predicted', fontsize=18)
plt.show()