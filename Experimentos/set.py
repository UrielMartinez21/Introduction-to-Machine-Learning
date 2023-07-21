import pandas as pd                                     #Pandas para arreglos
from sklearn import linear_model                        #Importamos el modelo lineal de sklearn
from sklearn.model_selection import train_test_split    #Area de entrenamiento y testing

reg = linear_model.LogisticRegression()                 #Modelo de regresion logistica


#------------Abrimos archivo CSV------------
iris= open("irisdatos.csv")
df = pd.read_csv(iris)
print(df)

#------------Area de etiquetas y caracteristicas------------
arreglox = df[df.columns[:-1]].to_numpy()  #Contendra las caracteristicas de los petalos
arregloy = df[df.columns[-1]].to_numpy()   #Contendra las etiquetas de los petalos

#print("[+]Caracteristicas de petalos")
#print(arreglox)

#print("[+]Etiquetas de petalos")
#print(arregloy)

#------------Area de entrenamiento y testing------------
X_train, X_test, y_train, y_test = train_test_split(arreglox, arregloy)
print(f"[+]Flores y caracteristicas: {X_train.shape}") 

reg.fit(X_train,y_train)

print(f"\n[+]El porcentaje es: {100*reg.score(X_test,y_test)}")
