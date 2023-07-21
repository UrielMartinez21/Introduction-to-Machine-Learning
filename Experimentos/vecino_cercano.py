#KNN(k-vecinos cercanos) = Algoritmo de tipo supervisado
#Se usa para clasificar muestras(valores discretos) o para predecir(regresion)

#sklearn = bibloteca popular de aprendizaje automatico
#sklearn = herramientas de clasificacion, regresion, agrupacion y reduccion de dimensionalidad

#----------------Area de importacion----------------
import numpy as np                                      #Para usar arreglos
from sklearn.datasets import load_iris                  #Datos de iris
from sklearn.model_selection import train_test_split    #Dividir informacion en set de entrenamiento y de testing

from sklearn.neighbors import KNeighborsClassifier      #Clasificador de vecinos cercanos

#----------------Area de cargar informacion----------------
iris=load_iris()                    #todos los sets de datos de iris
# print(type(iris))                 #bunch es un tipo de direccionario
# print(iris.keys())                #llaves de nuestro set de datos
# print(iris['data'])               #tiene todos los datos de los iris, renglon=flor, columna=medicion
# print(iris['target_names'])       #nombre de las flores
# print(iris['target'])             #conversion de los tipos de flores a numeros
# print(iris['feature_names'])      #nombre de las columnas de la data

#----------------Area de informacion----------------
flores = ",".join(iris['target_names'])
print(f"[+]Flores registradas: {flores}")

mediciones = ",".join(iris['feature_names'])
print(f"[+]Las mediciones son: {mediciones}\n")

#print(iris['target'])
#----------------Area de entrenamiento y testing----------------
#Separamos la informacion en set de entrenamiento y set de testing
#Regresara 4 valores
X_train,X_test,y_train,y_test=train_test_split(iris['data'],iris['target'])

#----------------Area de informacion----------------
print(f"[+]Flores y sus mediciones: {X_train.shape}")   #x para entrenar; numero de flores con sus 4 mediciones
#print(y_train.shape)                                   #vector que devolvera el numero de etiquetas

#----------------Area para predecir----------------
Knn=KNeighborsClassifier(n_neighbors=7)                                 #Considerar a los 7 vecinos mas cercanos
Knn.fit(X_train,y_train)                                                #Comando para entrenar 'fit'
print(f"[+]Porcentaje para predecir: {Knn.score(X_test, y_test)}")      #Que tan bien aprendio el algoritmo 
#print(Knn.predict([[1.2,3.4,5.6,1.1]]))                                #Predecir a que clasificacion acorde a sus medidas

def clasificar(valor):
    if valor==0:
        print("[+]Pertenece al grupo setosa")
    elif valor==1:
        print("[+]Pertenece al grupo versicolor")
    elif valor==2:
        print("[+]Pertenece al grupo virginia")
    else:
        print("[+]Se desconoce al grupo que pertenece")


clasificar(Knn.predict([[10, 3, 5, 1.2]]))
