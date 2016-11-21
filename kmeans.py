import numpy as np
from scipy.cluster.vq import kmeans, vq
from pylab import plot,show

data = np.loadtxt('iris.data' , delimiter=',', usecols=(0,1,2,3))
#data2 = np.loadtxt('iris.data' , delimiter=',' , usecols=(0,1))

#clusters , retorna distancia = kameans(base de dados e numero de centroids)
centroids,_= kmeans(data,3)
#print centroids  (coordenadas dos centroids)


#associar cada ponto aos centroids
idx,_ = vq(data,centroids)


#grafico

plot(data[idx==0,0],data[idx==0,1], 'ob',
	data[idx==1,0],data[idx==1,1], 'or',
	data[idx==2,0],data[idx==2,1], 'og')

plot(centroids[:,0],centroids[:,1],'sy',markersize=8)

show()