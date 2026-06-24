# main.py
from src.perceptron import Perceptron
from src.dataset import training_data

# Inicializar y entrenar el modelo
model = Perceptron()
model.train(training_data)

# Predecir con un cliente nuevo (Ingreso de 9, buen historial: 1)
prediction = model.predict([9, 1])
print(f"Predicción para el nuevo cliente [9, 1]: {prediction}")