#---------------| Importar librerias |-----------------#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

# ---------------| Tratamiento de dataset |-----------------#

# --> Importar dataset
resultado = pd.read_csv('50_Startups.csv')
resultado.head()

# --> Dividir dataset en variables independientes y dependientes
x = resultado.iloc[:, :-1].values
y = resultado.iloc[:, 4].values

# ---------------| Dividir dataset en prueba y entrenamiento |-----------------#

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2,random_state=0)

#---------------| Escalado de variables |-----------------#

from sklearn.preprocessing import StandardScaler

# --> Objeto para estandarizar
sc_x = StandardScaler()

# --> Aplicar estandarizacion
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# ---------------| Ajustar regresion con dataset(crear modelo) |-----------------#

from sklearn.linear_model import LinearRegression
regresion = LinearRegression()

#---------------| Predicción del modelo |-----------------#

y_pred = regresion.predict(x_test)

# ---------------| Visualizar los resultados |-----------------#

x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))

plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regresion.predict(x_grid), color = 'blue')

plt.title('Modelo de Regresión')
plt.xlabel('Posición del empleado')
plt.ylabel('Sueldo (en $)')

plt.show()
