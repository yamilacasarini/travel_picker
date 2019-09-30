import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

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

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.02)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(max_iter=100000)

# mlp.fit(X_test, y_test)

# predictions = mlp.predict(X_test)


parameter_space = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,),(10,10,10)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
clf.fit(X_test, y_test)

# All results
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))


# print(confusion_matrix(y_test, predictions))
# print(classification_report(y_test, predictions))



