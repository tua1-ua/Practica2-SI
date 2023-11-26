import random
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
import time


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

X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost()

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
    def fit(self, X, Y, verbose=False):
        n_samples, n_features = X.shape

        # Iniciar pesos de las observaciones a 1/n_observaciones
        w = np.full(n_samples, 1 / n_samples)

        self.clfs = []

        if verbose:
            print(f"Entrenando clasificador Adaboost para el dígito {Y[0]}, T={self.T}, A={self.A}")
            print("Entrenando clasificadores de umbral (con dimensión, umbral, dirección y error):")

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
                print(f"Añadido clasificador {t}: {best_clf.caracteristica}, {best_clf.umbral:.4f}, "
                      f"{'' if best_clf.polaridad == 1 else '-'}1, {min_error:.6f}")

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Calcular tasas de acierto
        y_train_pred = self.predict(X)
        y_test_pred = self.predict(X_test)

        accuracy_train = accuracy_score(Y, y_train_pred)
        accuracy_test = accuracy_score(Y_test, y_test_pred)

        if verbose:
            print(f"Tasas acierto (train, test) y tiempo: {accuracy_train * 100:.2f}%, {accuracy_test * 100:.2f}%, "
                  f"{elapsed_time:.3f} s.")

    ## Método para obtener una predicción con el clasificador fuerte Adaboost
    def predict(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)

        return y_pred


def entrenamiento(class_digit, T, A, verbose=False):
    # Cargar los datos de MNIST utilizando la función proporcionada
    X_train, y_train, X_test, y_test = load_MNIST_for_adaboost()

    # Filtrar para obtener solo la clase deseada
    mask_train = (y_train == class_digit)
    mask_test = (y_test == class_digit)

    X_train_class = X_train[mask_train]
    y_train_class = y_train[mask_train]
    X_test_class = X_test[mask_test]
    y_test_class = y_test[mask_test]

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
        np.vstack((X_train_class, X_test_class)),
        np.concatenate((y_train_class, y_test_class)),
        test_size=0.2,
        random_state=42
    )

    # Crear y entrenar el clasificador Adaboost con posibilidad de imprimir información detallada
    adaboost_classifier = Adaboost(T=T, A=A)
    adaboost_classifier.fit(X_train_split, y_train_split, verbose=verbose)

    # Hacer predicciones en conjuntos de entrenamiento y prueba
    y_train_pred = adaboost_classifier.predict(X_train_split)
    y_test_pred = adaboost_classifier.predict(X_test_split)

    # Calcular tasas de acierto
    accuracy_train = accuracy_score(y_train_split, y_train_pred)
    accuracy_test = accuracy_score(y_test_split, y_test_pred)

   

    
    
def main():
    entrenamiento(class_digit=5, T=5, A=20, verbose=True)


if __name__ == "__main__":
    main()