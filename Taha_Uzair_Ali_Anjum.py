import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
import time


def load_MNIST_for_adaboost(digit):
    # Cargar los datos de entrenamiento y test tal y como nos los sirve keras (MNIST de YannLecun)
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()

    # Formatear imágenes a vectores de floats y normalizar
    X_train = X_train.reshape((X_train.shape[0], 28*28)).astype("float32") / 255.0
    X_test = X_test.reshape((X_test.shape[0], 28*28)).astype("float32") / 255.0

    # Formatear las clases a enteros con signo para aceptar clase -1
    Y_train_bin = np.where(Y_train == digit, 1, -1)
    Y_test_bin = np.where(Y_test == digit, 1, -1)

    return X_train, Y_train_bin, X_test, Y_test_bin



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


def tareas_1A_y_1B_adaboost_binario(clase, T, A, verbose=False):
    X_entrenamiento, Y_entrenamiento, X_prueba, Y_prueba = load_MNIST_for_adaboost(clase)
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
    
     
if __name__ == "__main__":
    rend_1A = tareas_1A_y_1B_adaboost_binario(clase=9, T=180, A=20, verbose=True)