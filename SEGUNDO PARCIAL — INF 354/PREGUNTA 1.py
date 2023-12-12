#1. Generar una red neuronal (sin librerias) que utilice el dataset iris con producto punto, 
errores y de dos capas.

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

#función de activación
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#derivada de la función de activación
def sigmoid_derivative(x):
    return x * (1 - x)

#cargar el conjunto de datos Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

#normalizar los datos
X = X / np.amax(X, axis=0)
y = y.reshape(-1, 1)

#dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#inicializar pesos y bias aleatorios para las capas ocultas y de salida
input_neurons = X.shape[1]
hidden_neurons = 5
output_neurons = 1

np.random.seed(42)
hidden_weights = np.random.uniform(size=(input_neurons, hidden_neurons))
hidden_bias = np.random.uniform(size=(1, hidden_neurons))
output_weights = np.random.uniform(size=(hidden_neurons, output_neurons))
output_bias = np.random.uniform(size=(1, output_neurons))

#Hiperparámetros
epocas = 10000
learning_rate = 0.1

#Entrenamiento de la red neuronal
for epoca in range(epocas):
    #propagarcion hacia adelante
    hidden_layer_input = np.dot(X_train, hidden_weights) + hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, output_weights) + output_bias
    predicted_output = sigmoid(output_layer_input)

    #calculo de errores
    error = y_train - predicted_output

    #propagacion hacia atras
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    #actualizacion pesos y bias
    output_weights += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    hidden_weights += X_train.T.dot(d_hidden_layer) * learning_rate
    hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

#se evalua el modelo
hidden_layer_input = np.dot(X_test, hidden_weights) + hidden_bias
hidden_layer_output = sigmoid(hidden_layer_input)
output_layer_input = np.dot(hidden_layer_output, output_weights) + output_bias
predicted_output = sigmoid(output_layer_input)

#Redondeo de las predicciones a 0 o 1
predicted_output = np.round(predicted_output)

#Calcular la precision
accuracy = np.mean(predicted_output == y_test)
print(f"Precisión en el conjunto de prueba: {accuracy * 100:.2f}%")
