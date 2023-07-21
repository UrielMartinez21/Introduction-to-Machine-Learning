import joblib                   #Importamos la libreria que nos traera devuelta el algoritmo
from sklearn import datasets    #Datos de iris

#Mandamos a llamar el archivo que contiene el modelo
clf = joblib.load(
    r"C:\Users\Uriel Martinez\Desktop\Uriel Martinez\Personal\Programacion\Python\Machine Learning\modelo_entrenado.pkl")
iris=datasets.load_iris()                           #Cargamos los datos de iris a la variable iris

print(f"[+]Porcentaje de aprendizaje: {clf.score(iris['data'],iris['target'])}")    #Pasamos datos y etiquetas a clf

