import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from keras.datasets import mnist
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

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
    def __init__(self, digitos=10, T=5, A=20, verbose=False):
        self.digitos = digitos
        self.verbose = verbose
        self.classifiers = [Adaboost(T, A) for _ in range(digitos)]

    def fit(self, X, Y):
        for class_index, classifier in enumerate(self.classifiers):
            if self.verbose:
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
def tareas_1D_adaboost_multiclase(T, A, verbose=False):
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train.reshape((X_train.shape[0], -1)).astype('float32') / 255
    X_test = X_test.reshape((X_test.shape[0], -1)).astype('float32') / 255

    if verbose:
        print(f"\nComenzando el entrenamiento del clasificador Adaboost multiclase, T={T}, A={A}...")
    start_time = time.time()

    multi_class_clf = AdaboostMulticlase(digitos=10, T=T, A=A, verbose=verbose)
    multi_class_clf.fit(X_train, Y_train)

    end_time = time.time()
    elapsed_time = end_time - start_time

    predictions = multi_class_clf.predict(X_test)
    accuracy = np.mean(predictions == Y_test) * 100

    if verbose:
        print(f"Entrenamiento completado. Tasa de acierto: {accuracy:.2f}%. Tiempo total: {elapsed_time:.3f} s.")

    return accuracy



#########################################################################
# Adaboost multiclase de scikit-learn con los valor por defecto
######################################################################### 
def tarea_2A_AdaBoostClassifier_default(n_estimators=50):
    # Cargamos el dataset de mnist
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost()

    # Invocar el clasificador de scikit-learn
    adaboost_classifier = AdaBoostClassifier(n_estimators=n_estimators)

    # Imprimir mensaje de inicio del algoritmo
    print("Comenzando el AdaBoost multiclase de scikit-learn")

    # Medir el tiempo de ejecución
    start = time.time()

    # Entrenar el clasificador para el conjunto de entrenamiento mnist
    adaboost_classifier.fit(X_train, Y_train)

    end = time.time()
    t_ejec = end - start

    # Evaluar el modelo en el conjunto de prueba
    predictions = adaboost_classifier.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)

    # Imprimir la tasas de acierto y el tiempo de ejecución
    print(f'Tasa de acierto del clasificador AdaBoost de {n_estimators} estimadores: {accuracy}, tiempo de ejecución: {t_ejec:.3f} s.\n')

    return accuracy, t_ejec


#########################################################################
# Función que muestra las gráficas de rendimiento de los clasificadores
# de la tarea 1D y 2A (haciendo llamadas a funciones correspondientes)
######################################################################### 
def tarea_2B_graficas_rendimiento(bool_1D=True, bool_2A=True):
    if bool_1D:
        graficas_multiclase_1D()
    if bool_2A:
        graficas_multiclase_2A()


#########################################################################
# Esta función grafica la tasa de acierto y el tiempo de ejecución 
# para cada combinación de T y A del adaboost multiclase de la tarea 1D
######################################################################### 
def graficas_multiclase_1D():
    # Estos rangos los podemos ajustar en función de nuestras necesidades
    T_values = list(range(10, 90, 10))  # Valores de 10 a 90 con incrementos de 10
    A_values = list(range(10, 90, 10))  # Valores de 10 a 90 con incrementos de 10

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
        # Reutiliza la función de la tarea 1D para cada combinación de T y A
        result = tareas_1D_adaboost_multiclase(T=T, A=A, verbose=False)

        accuracies.append(result)
        # Supongamos un tiempo fijo para cada ejecución multiclase
        times.append(10.0)

    # Encontrar la mejor combinación según la tasa de acierto
    max_acc_index = np.argmax(accuracies)

    # Graficar la tasa de acierto en el primer eje Y
    ax1.plot(range(len(combinaciones_validas)), accuracies, color='tab:blue', marker='o', label='Tasa de Acierto')
    ax1.scatter(max_acc_index, accuracies[max_acc_index], color='green', marker='*', s=200, label='Mejor Tasa de Acierto')

    # Graficar el tiempo de ejecución en el segundo eje Y
    ax2.plot(range(len(combinaciones_validas)), times, color='tab:red', marker='s', label='Tiempo de Ejecución')
    ax2.scatter(np.argmin(times), np.min(times), color='blue', marker='*', s=200, label='Menor Tiempo de Ejecución')

    # Añadir leyendas y título
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title('Ajuste óptimo de T y A para el clasificador Adaboost Binario multiclase')
    
    # Modificar etiquetas del eje x
    ax1.set_xticks(range(len(combinaciones_validas)))
    ax1.set_xticklabels([f'{T}-{A}' for T, A in combinaciones_validas], rotation=45, ha='right')

    # Mostrar el valor de T y A correspondiente a la mejor tasa de acierto
    best_accuracy_combination = combinaciones_validas[max_acc_index]
    accuracy_info = f"Mejor tasa de acierto para {best_accuracy_combination}: Tasa de acierto: {accuracies[max_acc_index]:.2f}%, Tiempo de ejecución: {times[max_acc_index]:.3f} s"

    # Mostrar un solo print que incluye la información de los tres apartados
    print(accuracy_info)
    plt.show()
    
    
#########################################################################
# Esta función grafica la tasa de acierto y el tiempo de ejecución 
# para cada combinación de n_estimators del adaboost multiclase de la  2A
#########################################################################     
def graficas_multiclase_2A():
    n_estimators_values = list(range(10, 100, 10))  # Adjust the range based on your needs

    accuracies = []
    times = []

    for n_estimators in n_estimators_values:
        result = tarea_2A_AdaBoostClassifier_default(n_estimators=n_estimators) 
        accuracies.append(result[0])
        times.append(result[1])

    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Configurar el primer eje Y (tasa de acierto)
    ax1.set_xlabel('n_estimators')
    ax1.set_ylabel('Tasa de Acierto', color='tab:blue')
    ax1.plot(n_estimators_values, accuracies, color='tab:blue', marker='o', label='Tasa de Acierto')
    ax1.legend(loc='upper left')

    # Configurar el segundo eje Y (tiempo de ejecución)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Tiempo de Ejecución (s)', color='tab:red')
    ax2.plot(n_estimators_values, times, color='tab:red', marker='s', label='Tiempo de Ejecución')
    ax2.legend(loc='upper right')

    plt.title('Evolución de la Tasa de Acierto y Tiempo de Ejecución con n_estimators')
    plt.show()

    return accuracies, times

#########################################################################
# Versión optimizada del clasificador Adaboost multiclase de scikit-learn
#########################################################################  
def tarea_2A_AdaBoostClassifier_optimized(n_estimators=50, learning_rate=1.0, reduce_dimensionality=False, pca_components=None, n_jobs=None):
    # Cargamos el dataset de MNIST
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost()

    # Reducción de dimensionalidad si se especifica
    if reduce_dimensionality:
        pca = PCA(n_components=pca_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    # Configuración de AdaBoostClassifier con un clasificador débil personalizado
    base_classifier = DecisionTreeClassifier(max_depth=8, min_samples_split=4)

    # Configuración del clasificador débil para paralelización
    base_classifier.n_jobs = n_jobs

    adaboost_classifier = AdaBoostClassifier(base_classifier, n_estimators=n_estimators, learning_rate=learning_rate)

    # Entrenamiento y medición del tiempo de ejecución
    start = time.time()
    adaboost_classifier.fit(X_train, Y_train)
    end = time.time()
    t_ejec = end - start

    # Evaluación del modelo en el conjunto de prueba
    predictions = adaboost_classifier.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)

    # Imprimir resultados
    print(f'Tasa de acierto del clasificador AdaBoost con {n_estimators} estimadores y tasa de aprendizaje {learning_rate}: {accuracy:.4f}, tiempo de ejecución: {t_ejec:.3f} s.\n')


#########################################################################
# Función que construye un modelo MLP con Keras
######################################################################### 
def build_mlp_model(input_shape, num_classes, hidden_layers=1, neurons_per_layer=128, activation='relu', output_activation='softmax', optimizer='adam', learning_rate=0.001):
    model = Sequential()
    
    model.add(Dense(neurons_per_layer, activation=activation, input_shape=(input_shape,)))
    model.add(BatchNormalization())  # Agregamos normalización por lotes

    for _ in range(hidden_layers - 1):
        model.add(Dense(neurons_per_layer, activation=activation))
        model.add(BatchNormalization())

    model.add(Dense(num_classes, activation=output_activation))

    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

#########################################################################
# Clasicador MLP 
######################################################################### 
def tarea_2D_AdaBoostClassifier_DecisionTree(hidden_layers=1, neurons_per_layer=128, activation='relu', output_activation='softmax', optimizer='adam', learning_rate=0.001):
    # Cargar los datos de MNIST
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost()

    # Convertir las etiquetas a categorías
    Y_train_cat = to_categorical(Y_train, num_classes=10)
    Y_test_cat = to_categorical(Y_test, num_classes=10)

    # Dividir datos de entrenamiento para validación
    X_train, X_val, Y_train_cat, Y_val_cat = train_test_split(X_train, Y_train_cat, test_size=0.2, random_state=42)

    input_shape = X_train.shape[1]
    num_classes = Y_train_cat.shape[1]

    # Construir el modelo MLP
    model = build_mlp_model(input_shape, num_classes, hidden_layers, neurons_per_layer, activation, output_activation, optimizer, learning_rate)

    # Entrenar el modelo
    start_time = time.time()
    model.fit(X_train, Y_train_cat, epochs=10, batch_size=32, validation_data=(X_val, Y_val_cat), verbose=2)
    end_time = time.time()

    # Evaluar el modelo
    test_loss, test_accuracy = model.evaluate(X_test, Y_test_cat, verbose=0)
    
    print(f"Exactitud en los datos de prueba: {test_accuracy * 100:.2f}%")
    print(f"Tiempo de ejecución: {end_time - start_time:.3f} s\n")

    return test_accuracy

#########################################################################
# Classificador CNN redes neuronales convolucionales
######################################################################### 
def tarea_2E_CNN_Keras(n_conv_layers, n_filters, filter_size, n_dense_layers, n_dense_neurons):
    # Cargar datos de MNIST
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    # Preprocesamiento de datos
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32') / 255
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32') / 255

    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)

    # Dividir datos de entrenamiento para validación
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

    # Construir el modelo CNN
    model = Sequential()

    # Capas convolucionales
    for _ in range(n_conv_layers):
        model.add(Conv2D(n_filters, (filter_size, filter_size), activation='relu', input_shape=(28, 28, 1)))
        model.add(MaxPooling2D((2, 2)))

    # Capas totalmente conectadas
    model.add(Flatten())
    for _ in range(n_dense_layers):
        model.add(Dense(n_dense_neurons, activation='relu'))

    model.add(Dense(10, activation='softmax'))  # Capa de salida

    # Compilar el modelo
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Entrenar el modelo
    start_time = time.time()
    model.fit(X_train, Y_train, epochs=10, batch_size=64, validation_data=(X_val, Y_val), verbose=2)
    end_time = time.time()

    # Evaluar el modelo
    test_loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=0)

    print(f"Exactitud en los datos de prueba: {test_accuracy * 100:.2f}%")
    print(f"Tiempo de ejecución: {end_time - start_time:.3f} s\n")

    return test_accuracy


#########################################################################
# main para ejecutar las tareas
######################################################################### 
if __name__ == "__main__":
    rend_1A = tareas_1A_y_1B_adaboost_binario(clase=9, T=45, A=10, verbose=True)
    #tarea_1C_graficas_rendimiento()
    #rend_1D = tareas_1D_adaboost_multiclase(T=100, A=30, verbose=True)
    #rend_2A = tarea_2A_AdaBoostClassifier_default(n_estimators=40) # la configuración óptima
    #tarea_2B_graficas_rendimiento(bool_1D=False, bool_2A=True)
    #rend_2B= tarea_2A_AdaBoostClassifier_optimized(n_estimators=10, learning_rate=0.1, reduce_dimensionality=True, pca_components=20, n_jobs=-1)
    #rend_2D = tarea_2D_AdaBoostClassifier_DecisionTree(hidden_layers=2, neurons_per_layer=256, learning_rate=0.01)
    #rend_2E_CNN = tarea_2E_CNN_Keras(n_conv_layers=2, n_filters=32, filter_size=3, n_dense_layers=2, n_dense_neurons=128)

