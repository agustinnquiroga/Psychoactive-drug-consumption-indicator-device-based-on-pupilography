import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
import joblib
import graphviz 
import pydot
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

datos = pd.read_csv('sujetos_control-thc_positivos.csv', sep=';') # SE CARGAN LOS DATOS 

##### SE VISUALIZAN LOS DATOS #####

fig, ax = plt.subplots(figsize=(8,5))
plt.grid()
ax.set_axisbelow(True)
sin_droga = datos.drop(columns = 'test')
sin_droga = sin_droga.truncate(before=0, after=9)
ax.scatter(sin_droga.tiempo_respuesta, sin_droga.area_relativa, c="darkblue",s=30);
con_droga = datos.drop(columns = 'test')
con_droga = con_droga.truncate(before=10,after=14)
ax.scatter(con_droga.tiempo_respuesta, con_droga.area_relativa, c="red",s=30);
ax.set_xlim([0.4, 1.05])
ax.set_ylim([35, 70])
ax.set_xticks([0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05])
ax.set_xlabel("Tiempo de Respuesta")
ax.set_ylabel("Amplitud relativa")

##### CONFIGURACIÓN MODELO #####

datos_con_y_sin = datos.drop(columns = 'test')
datos_con_y_sin = datos_con_y_sin.truncate(before=0, after=14)
clasificador = datos['test']
clasificador = clasificador.truncate(before=0, after=14)
modelo = tree.DecisionTreeClassifier(max_leaf_nodes = 10)
modelo.fit(datos_con_y_sin, clasificador)
dot_data = tree.export_graphviz(modelo, out_file=None, 
                         feature_names=datos_con_y_sin.columns.values,  
                         class_names=['Nada','THC'],
                         filled=True, rounded=True,  
                         special_characters=True,leaves_parallel = False)  
graph = graphviz.Source(dot_data)  
graph 
(graph, ) = pydot.graph_from_dot_data(dot_data)

graph.write_png('arbol_ia.png') # GUARDA ARBOL DE DECISIÓN

joblib.dump(modelo,"modelo.pkl")

##### TESTEO #####

X_test = datos_con_y_sin.truncate(before=0, after=14)
print(X_test)
y_test = clasificador.truncate(before=0, after=14)
print(modelo.score(X_test, y_test)) # PUNTAJE DE PREDICCIÓN
y_pred = modelo.predict(X_test)
matriz = confusion_matrix(y_test,y_pred)
plot_confusion_matrix(conf_mat=matriz, figsize=(6,6), show_normed=False)
plt.tight_layout()


prueba = datos = pd.read_csv('datos_alcohol.csv', sep=';')
prueba= prueba.drop(columns = 'Sujeto de Prueba' )
prueba= prueba.drop(columns = 'droga' )

print(prueba)

resultado = modelo.predict(prueba)
print(resultado)