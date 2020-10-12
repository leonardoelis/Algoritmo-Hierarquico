import pandas as pd
import scipy.cluster.hierarchy as sch
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering

dataset = pd.read_csv("Mall_Customers.csv")
dados = dataset[['Annual Income (k$)', 'Spending Score (1-100)']].values

h = sch.linkage(dados, method='ward')
dendrogram = sch.dendrogram(h)
plt.show()

model = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
model.fit(dados)
rotulos = model.labels_
#print(rotulos)

plt.scatter(dados[rotulos == 0, 0], dados[rotulos == 0, 1], s=50, color='red', label='Cluster 1')
plt.scatter(dados[rotulos == 1, 0], dados[rotulos == 1, 1], s=50, color='blue', label='Cluster 2')
plt.scatter(dados[rotulos == 2, 0], dados[rotulos == 2, 1], s=50, color='green', label='Cluster 3')
plt.scatter(dados[rotulos == 3, 0], dados[rotulos == 3, 1], s=50, color='purple', label='Cluster 4')
plt.scatter(dados[rotulos == 4, 0], dados[rotulos == 4, 1], s=50, color='orange', label='Cluster 5')
plt.legend()
plt.show()

