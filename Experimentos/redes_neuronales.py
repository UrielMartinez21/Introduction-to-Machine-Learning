#Emular el cerebro humano
#Nodos entre la maquina y que esten comunicados entre ellos

#                   y=sum(w*x)+b
# w=peso(importancia)     x=valor     b(sesgo)=manera de decir si es propensa a activarse o no
#Ejemplo: y se activa si (w*x)+b>0.5

#Entradas-Capas_Ocultas-Salidas
#Cada nodo tiene un peso y un valor el cual manda a la sigiente capa

#-------------Area de importacion de librerias-------------
import sklearn
from sklearn.datasets import load_iris                  #Importamos nuestro set de datos de iris
from sklearn.model_selection import train_test_split    #Dividimos nuestro set en testing y entrenamiento
from sklearn.neural_network import MLPClassifier        #Clasificador de redes neuronales

iris=load_iris()                                        #Todos los datos del set se lo pasamos a la variable iris
caract=iris.data                                        #Caracteristicas
etiq=iris.target                                        #Etiquetas

#----------------Area de entrenamiento y testing----------------
X_train,X_test,y_train,y_test=train_test_split(caract,etiq)

#max_iter= proceso donde se mandan informacion ocurrira 10 veces
#hidden_layer_sizes= 1 capa que tendra solo 10 nodos
#Mientras menor sea el numero de iteraciones menor sera el numero de predicciones correctas
red=MLPClassifier(max_iter=1000, hidden_layer_sizes=(10))

#hidden_layer_sizes; 10 capaz y cada capa tendra 5 nodos
#red = MLPClassifier(max_iter=10000, hidden_layer_sizes=(10,5))

red.fit(X_train,y_train)                                        #Le pasamos nuestras variables de entrenamiento
print(f"[+]Porcentaje de acierto: {100*red.score(X_test,y_test)}")  #Ver el porcentaje de predicciones que tuve