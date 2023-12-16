import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
import time
from keras.datasets import mnist
from mpl_toolkits.mplot3d import Axes3D


#########################################################################
# Cargamos el dataset para un dígito en específico
######################################################################### 
def load_MNIST_for_adaboost_dig(digit):
    # Cargar los datos de entrenamiento y test tal y como nos los sirve keras (MNIST de YannLecun)
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()

    # Formatear imágenes a vectores de floats y normalizar
    X_train = X_train.reshape((X_train.shape[0], 28*28)).astype("float32") / 255.0
    X_test = X_test.reshape((X_test.shape[0], 28*28)).astype("float32") / 255.0

    # Formatear las clases a enteros con signo para aceptar clase -1
    Y_train_bin = np.where(Y_train == digit, 1, -1)
    Y_test_bin = np.where(Y_test == digit, 1, -1)

    return X_train, Y_train_bin, X_test, Y_test_bin


#########################################################################
# Cargamos el dataset para todos los dígitos
######################################################################### 
def load_MNIST_for_adaboost():
    # Cargar los datos de entrenamiento y test tal y como nos los sirve keras (MNIST de YannLecun)
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
    # Formatear imágenes a vectores de floats y normalizar
    X_train = X_train.reshape((X_train.shape[0], 28*28)).astype("float32") / 255.0
    X_test = X_test.reshape((X_test.shape[0], 28*28)).astype("float32") / 255.0
    #X_train = X_train.astype("float32") / 255.0
    #X_test = X_test.astype("float32") / 255.0
    # Formatear las clases a enteros con signo para aceptar clase -1
    Y_train = Y_train.astype("int8")
    Y_test = Y_test.astype("int8")

    return X_train, Y_train, X_test, Y_test


#########################################################################
# La clase DecisionStump (clasificador débil)
######################################################################### 
class DecisionStump:
    ## Constructor de clase, con número de características
    def __init__(self, n_features):
        # Seleccionar al azar una característica, un umbral y una polaridad.
        self.caracteristica = np.random.randint(0, n_features)
        self.umbral = np.random.rand()
        self.polaridad = np.random.choice([-1, 1])
        
        
    ## Método para obtener una predicción con el clasificador débil
    def predict(self, X):
        X_column = X[:, self.caracteristica]
        # Inicializamos un arreglo de predicciones con todos los elementos como 1.
        predictions = np.ones(X.shape[0])
        # Actualizamos las predicciones a -1 cuando carac*pol<umbral*pol
        predictions[(self.polaridad * X_column) < (self.polaridad * self.umbral)] = -1
        return predictions


#########################################################################
# La clase Adaboost (clasificador fuerte)
######################################################################### 
class Adaboost:
    ## Constructor de clase, con número de clasificadores e intentos por clasificador
    def __init__(self, T=5, A=20):
        # Dar valores a los parámetros del clasificador e iniciar la lista de clasificadores débiles vacía
        self.T = T
        self.A = A
        self.clfs = []

    ## Método para entrenar un clasificador fuerte a partir de clasificadores débiles mediante Adaboost
    ## Método para entrenar un clasificador fuerte a partir de clasificadores débiles mediante Adaboost
    def fit(self, X, Y, verbose=False):  # Modified to include test set
        n_samples, n_features = X.shape

        # Iniciar pesos de las observaciones a 1/n_observaciones
        w = np.full(n_samples, 1 / n_samples)

        self.clfs = []

        # Bucle de entrenamiento Adaboost: desde 1 hasta T repetir
        start_time = time.time()
        for t in range(1, self.T + 1):
        
            best_clf = None
            min_error = float("inf")

            # Bucle de búsqueda de un buen clasificador débil: desde 1 hasta A repetir
            for a in range(1, self.A + 1):
                # Crear un nuevo clasificador débil aleatorio
                clf = DecisionStump(n_features)
                predictions = clf.predict(X)

                # Calcular el error: comparar predicciones con los valores deseados
                error = np.sum(w * (predictions != Y))

                if error < min_error:
                    best_clf = clf
                    min_error = error

            # Calcular el valor de alfa y las predicciones del mejor clasificador débil
            EPS = 1e-10
            best_clf.alpha = 0.5 * np.log((1.0 - min_error + EPS) / (min_error + EPS))
            best_predictions = best_clf.predict(X)

            # Actualizar pesos de las observaciones en función de las predicciones, los valores deseados y alfa
            w *= np.exp(-best_clf.alpha * Y * best_predictions)
            # Normalizar a 1 los pesos
            w /= np.sum(w)

            # Guardar el clasificador en la lista de clasificadores de Adaboost
            self.clfs.append(best_clf)

            if verbose:
                print(f"Añadido el clasificador {t}: {best_clf.caracteristica}, {best_clf.umbral:.4f}, "
                      f"{'' if best_clf.polaridad == 1 else '-'}1, {min_error:.6f}")

        end_time = time.time()
        elapsed_time = end_time - start_time

        
   
    ## Método para obtener una predicción con el clasificador fuerte Adaboost
    def predict(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)
        return y_pred


#########################################################################
# La clase AdaboostMulticlase (multiclasificador fuerte-no óptimo)
#########################################################################     
class AdaboostMulticlase:
    def __init__(self, digitos=10, T=5, A=20):
        self.digitos = digitos
        self.classifiers = [Adaboost(T, A) for _ in range(digitos)]

    def fit(self, X, Y):
        for class_index, classifier in enumerate(self.classifiers):
            print(f"Entrenando el clasificador multiclase para la clase {class_index}")
            binary_labels = self._create_binary_labels(Y, class_index)
            classifier.fit(X, binary_labels, verbose=False)

    def predict(self, X):
        predictions = np.array([classifier.predict(X) for classifier in self.classifiers])
        normalized_predictions = self._normalize_predictions(predictions)
        return np.argmax(normalized_predictions, axis=0)

    def _create_binary_labels(self, Y, class_index):
        return np.where(Y == class_index, 1, -1)

    def _normalize_predictions(self, predictions):
        min_values = predictions.min(axis=0)
        max_values = predictions.max(axis=0)
        return (predictions - min_values) / (max_values - min_values + 1e-8)


#########################################################################
# La clase AdaboostMulticlaseMejorado (multiclasificador fuerte-óptimo)
#########################################################################  
class AdaboostMulticlaseMejorado:
    def __init__(self, digitos=10, T=5, A=20):
        self.digitos = digitos
        self.classifiers = [Adaboost(T, A) for _ in range(digitos)]

    def fit(self, X, Y):
        for class_index, classifier in enumerate(self.classifiers):
            print(f"Entrenando el clasificador multiclase para la clase {class_index}")
            binary_labels = self._create_binary_labels(Y, class_index)
            classifier.fit(X, binary_labels, verbose=False)

    def predict(self, X):
        predictions = np.array([classifier.predict(X) for classifier in self.classifiers])
        normalized_predictions = (predictions - predictions.min(axis=0)) / (predictions.max(axis=0) - predictions.min(axis=0) + 1e-8)
        return np.argmax(normalized_predictions, axis=0)

    def _create_binary_labels(self, Y, class_index):
        return np.where(Y == class_index, 1, -1)


#########################################################################
# Método para probar el rendimiento del Adaboost (tarea 1A y 1B)
#########################################################################  
def tareas_1A_y_1B_adaboost_binario(clase, T, A, verbose=False):
    X_entrenamiento, Y_entrenamiento, X_prueba, Y_prueba = load_MNIST_for_adaboost_dig(clase)
    adaboost = Adaboost(T=T, A=A)

    inicio = time.time() 

    if verbose:
        print(f"Entrenando clasificador Adaboost para el dígito {clase}, T={T}, A={A}")
        print("Entrenando clasificadores de umbral (con dimensión, umbral, dirección y error):")

    adaboost.fit(X_entrenamiento, Y_entrenamiento, verbose=verbose)

    fin = time.time()

    entrenamiento_predicciones = adaboost.predict(X_entrenamiento)
    prueba_predicciones = adaboost.predict(X_prueba)

    aciertos_entrenamiento = np.mean(entrenamiento_predicciones == Y_entrenamiento) * 100
    aciertos_test = np.mean(prueba_predicciones == Y_prueba) * 100
    t_ejec = fin - inicio

    if verbose:
        print(f"Tasas acierto (train, test) y tiempo: {aciertos_entrenamiento:.2f}%, {aciertos_test:.2f}%, {t_ejec:.3f} s.")

    return {"aciertos_entrenamiento": aciertos_entrenamiento, "aciertos_test": aciertos_test, "tiempo_ejecución": t_ejec}


#########################################################################
# Método para graficar el rendimiento del Adaboost (tarea 1C)
#########################################################################  
def calculate_score(accuracy, time, weight_accuracy=0.8, weight_time=0.2):
    # Calcular la puntuación como una combinación ponderada de la tasa de acierto y el tiempo de ejecución
    score = weight_accuracy * accuracy - weight_time * time
    return score

def find_best_combination(combinaciones_validas, get_score_fn):
    # Obtener puntuaciones para cada combinación
    scores = [get_score_fn(T, A) for T, A in combinaciones_validas]

    # Encontrar la posición del máximo en la lista de puntuaciones
    max_score_index = np.argmax(scores)

    return combinaciones_validas[max_score_index]

def tarea_1C_graficas_rendimiento():
    # Estos rangos los podemos ajustar en función de nuestras necesidades
    T_values = list(range(5, 180, 10))  # Valores de 5 a 30 con incrementos de 5
    A_values = list(range(5, 30, 5))  # Valores de 5 a 180 con incrementos de 10

    # Filtrar combinaciones válidas según la restricción T * A ≤ 900
    combinaciones_validas = [(T, A) for T in T_values for A in A_values if T * A <= 900]

    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Configurar el primer eje Y (tasa de acierto)
    ax1.set_xlabel('Combinaciones T-A')
    ax1.set_ylabel('Tasa de Acierto', color='tab:blue')

    # Configurar el segundo eje Y (tiempo de ejecución)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Tiempo de Ejecución (s)', color='tab:red')

    accuracies = []  # Mover fuera del bucle
    times = []  # Mover fuera del bucle

    for T, A in combinaciones_validas:
        # Reutiliza la función de la tarea 1A para cada combinación de T y A
        result = tareas_1A_y_1B_adaboost_binario(clase=5, T=T, A=A)

        accuracies.append(result["aciertos_test"])
        times.append(result["tiempo_ejecución"])

    # Encontrar la mejor combinación según la tasa de acierto
    max_acc_index = np.argmax(accuracies)

    # Encontrar la mejor combinación según el tiempo de ejecución
    min_time_index = np.argmin(times)

    # Graficar la tasa de acierto en el primer eje Y
    ax1.plot(range(len(combinaciones_validas)), accuracies, color='tab:blue', marker='o', label='Tasa de Acierto')
    ax1.scatter(max_acc_index, accuracies[max_acc_index], color='green', marker='*', s=200, label='Mejor Tasa de Acierto')

    # Graficar el tiempo de ejecución en el segundo eje Y
    ax2.plot(range(len(combinaciones_validas)), times, color='tab:red', marker='s', label='Tiempo de Ejecución')
    ax2.scatter(min_time_index, times[min_time_index], color='blue', marker='*', s=200, label='Menor Tiempo de Ejecución')

    # Añadir leyendas y título
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title('Evolución de la Tasa de Acierto y Tiempo de Ejecución con T y A')
    
    # Modificar etiquetas del eje x
    ax1.set_xticks(range(len(combinaciones_validas)))
    ax1.set_xticklabels([f'{T}-{A}' for T, A in combinaciones_validas], rotation=45, ha='right')

    # Mostrar el valor de T y A correspondiente a la mejor tasa de acierto
    #print(f"Mejor tasa de acierto para: {combinaciones_validas[max_acc_index]}")

    # Mostrar el valor de T y A correspondiente a la mejor puntuación
    best_combination = find_best_combination(combinaciones_validas, get_score_fn=lambda T, A: calculate_score(accuracies[-1], times[-1]))
    #print(f"Mejor puntuación para: {best_combination}")

    # Mostrar el valor de T y A correspondiente al menor tiempo de ejecución
    #print(f"Mejor tiempo de ejecución para: {combinaciones_validas[min_time_index]}")


    # Mostrar la tasa de acierto y el tiempo de ejecución para la mejor tasa de acierto
    best_accuracy_combination = combinaciones_validas[max_acc_index]
    accuracy_info = f"Mejor tasa de acierto para {best_accuracy_combination}: Tasa de acierto: {accuracies[max_acc_index]:.2f}%, Tiempo de ejecución: {times[max_acc_index]:.3f} s"

    # Mostrar la tasa de acierto y el tiempo de ejecución para la mejor puntuación
    best_score_info = f"Mejor puntuación para {best_combination}: Tasa de acierto: {accuracies[-1]:.2f}%, Tiempo de ejecución: {times[-1]:.3f} s"

    # Mostrar la tasa de acierto y el tiempo de ejecución para el menor tiempo de ejecución
    best_time_combination = combinaciones_validas[min_time_index]
    min_time_info = f"Mejor tiempo de ejecución para {best_time_combination}: Tasa de acierto: {accuracies[min_time_index]:.2f}%, Tiempo de ejecución: {times[min_time_index]:.3f} s"

    # Mostrar un solo print que incluye la información de los tres apartados
    print(accuracy_info)
    print(best_score_info)
    print(min_time_info)
    plt.show()



#########################################################################
# Método para mostrar rendimiento del Adaboost multiclase (tarea 1D)
#########################################################################  
def tareas_1D_adaboost_multiclase(T, A):
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train.reshape((X_train.shape[0], -1)).astype('float32') / 255
    X_test = X_test.reshape((X_test.shape[0], -1)).astype('float32') / 255

    print(f"\nComenzando el entrenamiento del clasificador Adaboost multiclase, T={T}, A={A}...")
    start_time = time.time()

    multi_class_clf = AdaboostMulticlase(digitos=10, T=T, A=A)
    multi_class_clf.fit(X_train, Y_train)

    end_time = time.time()
    elapsed_time = end_time - start_time

    predictions = multi_class_clf.predict(X_test)
    accuracy = np.mean(predictions == Y_test) * 100

    print(f"Entrenamiento completado. Tasa de acierto: {accuracy:.2f}%. Tiempo total: {elapsed_time:.3f} s.")

    return accuracy


#########################################################################
# Método que mostrar rendimiento de Adaboost multiclase óptimo (tarea 1D)
######################################################################### 
def tarea_1E_adaboost_multiclase_mejorado(T, A):
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train.reshape((X_train.shape[0], -1)).astype('float32') / 255
    X_test = X_test.reshape((X_test.shape[0], -1)).astype('float32') / 255

    print(f"\nComenzando el entrenamiento del clasificador Adaboost multiclase mejorado, T={T}, A={A}...")
    start_time = time.time()

    multi_class_clf = AdaboostMulticlaseMejorado(digitos=10, T=T, A=A)
    multi_class_clf.fit(X_train, Y_train)

    end_time = time.time()
    elapsed_time = end_time - start_time

    predictions = multi_class_clf.predict(X_test)
    accuracy = np.mean(predictions == Y_test) * 100

    print(f"Entrenamiento completado. Tasa de acierto: {accuracy:.2f}%. Tiempo total: {elapsed_time:.3f} s.")

    return accuracy



if __name__ == "__main__":
    #rend_1A = tareas_1A_y_1B_adaboost_binario(clase=9, T=45, A=10, verbose=True)
    #tarea_1C_graficas_rendimiento()
    #rend_1D = tareas_1D_adaboost_multiclase(T=100, A=30)
    rend_1E = tarea_1E_adaboost_multiclase_mejorado(T=100, A=30)