#Persistencia del modelo
#Guardar un algoritmo entrenado para usar en otros archivos
#Hay modelos que tardan hasta semanas en entrenarse
#Se llama para no volver a entrenarlo en un futuro

#----------------Area de importancion----------------
from sklearn import datasets,linear_model               #Traemos los datos de iris y se sacara el modelo de regresion
from sklearn.model_selection import train_test_split    #Dividimos nuestro set en entrenamiento y testing

#----------------Area de declaracion----------------
iris=datasets.load_iris()                               #Cargamos los datos de iris en la variable iris
clf=linear_model.LogisticRegression()                   #Nuestra variable clasificador

#print(iris.keys())                                     #Revisar informacion como esta ordenada

#----------------Area de entrenamiento y testing----------------
X_train,X_test,y_train,y_test=train_test_split(iris.data,iris['target'])

clf.fit(X_train,y_train)                                            #Algoritmo entrenador
print(f"[+]Porcentaje de aprendizaje: {clf.score(X_test,y_test)}")  #Ver que tan bien aprendio el algoritmo 

#----------------Area de salvar algoritmo----------------
import joblib                                       #Libreria para guardar nuestro modelo entrenado

#clf contiene nuestro algoritmo ya entrenado
#Nombre con el que se guardara el archivo
joblib.dump(clf,"modelo_entrenado.pkl")
