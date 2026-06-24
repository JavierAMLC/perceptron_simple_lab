# tests/test_perceptron.py
from src.perceptron import Perceptron
from src.dataset import training_data

def test_prediction():
    model = Perceptron()
    model.train(training_data)
    # Debería aprobar (1) para un cliente solvente
    assert model.predict([8, 1]) == 1

def test_rejection():
    model = Perceptron()
    model.train(training_data)
    # Debería rechazar (0) para un cliente insolvente
    assert model.predict([1, 0]) == 0