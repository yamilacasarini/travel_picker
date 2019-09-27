import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

dataUrl = "./resources/setInicial.csv"

columnsName = ['ciudad_destino_anterior_compra', 'frecuencia_viajes_x_anio', 'ingresos_a_la_pagina_x_anio',
               'dias_anticipacion_anterior_compra', 'cantidad_pasajeros_anterior_compra',
               'duracion_viaje_anterior_compra', 'ciudad_destino']

data = pd.read_csv(dataUrl, names=columnsName, sep='|')

x = data.iloc[:, 0:5]

y = data.iloc[:, 6:7]

le = preprocessing.LabelEncoder()

y = le.fit_transform(y.values.ravel())

x.ciudad_destino_anterior_compra = le.fit_transform(np.ravel(x.ciudad_destino_anterior_compra))
x.frecuencia_viajes_x_anio = le.fit_transform(np.ravel(x.frecuencia_viajes_x_anio))
x.ingresos_a_la_pagina_x_anio = le.fit_transform(np.ravel(x.ingresos_a_la_pagina_x_anio))

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.005)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=10000)

mlp.fit(X_test, y_test)

predictions = mlp.predict(X_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))



