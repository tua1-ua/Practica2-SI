import random
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras


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


class DecisionStump:
    ## Constructor de clase, con número de características
    def __init__(self, n_features):
        # Seleccionar al azar una característica, un umbral y una polaridad.
        self.caracteristica = random.randint(0, n_features - 1)
        self.umbral = random.uniform(0, 1)
        self.polaridad = random.choice([-1, 1])
        
        
    ## Método para obtener una predicción con el clasificador débil
    def predict(self, X):
        predictions = np.where(X[:, self.caracteristica] * self.polaridad > self.umbral * self.polaridad, 1, -1)
        return predictions
        # Si la característica que comprueba este clasificador es mayor que el umbral y la polaridad es 1
        # o si es menor que el umbral y la polaridad es -1, devolver 1 (pertenece a la clase)
        # Si no, devolver -1 (no pertenece a la clase)




class Adaboost:
    def __init__(self, T=5, A=20):
        self.T = T
        self.A = A
        self.classifiers = []

    def fit(self, X, Y, verbose=False):
        n_samples, n_features = X.shape
        w = np.ones(n_samples) / n_samples

        for _ in range(self.T):
            best_classifier = None
            best_error = float('inf')

            for _ in range(self.A):
                classifier = DecisionStump(n_features)
                predictions = classifier.predict(X)
                error = np.sum(w * (predictions != Y))

                if error < best_error:
                    best_error = error
                    best_classifier = classifier

            alpha = 0.5 * np.log((1 - best_error) / (best_error + 1e-10))
            self.classifiers.append((best_classifier, alpha))

            predictions = best_classifier.predict(X)
            factor = np.exp(-alpha * Y * predictions)
            w = w * factor
            w = w / np.sum(w)

            if verbose:
                print(f"Añadido clasificador: {best_classifier.caracteristica}, {best_classifier.umbral}, {best_classifier.polaridad}, {best_error}")

    def predict(self, X):
        final_predictions = np.zeros(X.shape[0])
        for classifier, alpha in self.classifiers:
            final_predictions += alpha * classifier.predict(X)

        return np.sign(final_predictions)

# Tu implementación del DecisionStump aquí

# Ejemplo de uso
if __name__ == "__main__":
    # Cargar datos de MNIST y dividir en conjunto de entrenamiento y prueba
    # X_train, Y_train, X_test, Y_test = cargar_datos_mnist()
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost()

    # Crear y entrenar el clasificador Adaboost
    adaboost = Adaboost(T=20, A=10)
    adaboost.fit(X_train, Y_train, verbose=True)

    # Realizar predicciones en el conjunto de entrenamiento y prueba
    train_predictions = adaboost.predict(X_train)
    test_predictions = adaboost.predict(X_test)

    # Calcular tasas de acierto
    train_accuracy = np.mean(train_predictions == Y_train)
    test_accuracy = np.mean(test_predictions == Y_test)

    print(f"Tasa de acierto en entrenamiento: {train_accuracy * 100:.2f}%")
    print(f"Tasa de acierto en prueba: {test_accuracy * 100:.2f}%")