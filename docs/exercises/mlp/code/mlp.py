"""MLP em NumPy, com número de camadas definido por layer_sizes.

As funções da saída e da perda são argumentos para permitir o reuso do
mesmo forward, backward e update em classificação binária ou multiclasse.
"""

import numpy as np


def sigmoid(z):
    """Sigmoid calculada por partes para evitar overflow nas exponenciais."""
    probabilities = np.empty_like(z, dtype=float)
    positive = z >= 0
    probabilities[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exponential = np.exp(z[~positive])
    probabilities[~positive] = exponential / (1.0 + exponential)
    return probabilities


def binary_cross_entropy(logits, probabilities, y):
    """Retorna a BCE média e sua derivada em relação aos logits.

    log(1 + exp(z)) - y*z é equivalente à BCE com sigmoid, mas evita
    log(0). A derivada (p-y)/N já inclui a média sobre as amostras.
    """
    targets = np.asarray(y, dtype=float).reshape(-1, 1)
    loss = np.mean(np.logaddexp(0.0, logits) - targets * logits)
    delta = (probabilities - targets) / len(targets)
    return float(loss), delta


class MLP:
    """Camadas ocultas tanh e saída/perda configuráveis, sem autodiff."""

    def __init__(self, layer_sizes, rng, output_activation=sigmoid,
                 loss_and_delta=binary_cross_entropy):
        self.layer_sizes = list(layer_sizes)
        if len(self.layer_sizes) < 3 or any(size < 1 for size in self.layer_sizes):
            raise ValueError("Use entrada, pelo menos uma camada oculta e saída.")
        self.output_activation = output_activation
        self.loss_and_delta = loss_and_delta
        self.weights = []
        self.biases = []
        for n_in, n_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            # Inicialização de Xavier: escala adequada para a ativação tanh.
            scale = np.sqrt(2.0 / (n_in + n_out))
            self.weights.append(rng.normal(0.0, scale, size=(n_in, n_out)))
            self.biases.append(np.zeros((1, n_out)))

    def forward(self, X):
        """Amostras nas linhas; guarda as ativações para o backward."""
        activations = [np.asarray(X, dtype=float)]
        for layer, (weights, bias) in enumerate(zip(self.weights, self.biases)):
            logits = activations[-1] @ weights + bias
            if layer == len(self.weights) - 1:
                activation = self.output_activation(logits)
            else:
                activation = np.tanh(logits)
            activations.append(activation)
        return activations[-1], activations, logits

    def backward(self, activations, delta):
        """Deriva todos os pesos e biases antes de alterar os parâmetros."""
        grad_weights = [None] * len(self.weights)
        grad_biases = [None] * len(self.biases)
        for layer in range(len(self.weights) - 1, -1, -1):
            grad_weights[layer] = activations[layer].T @ delta
            grad_biases[layer] = delta.sum(axis=0, keepdims=True)
            if layer > 0:
                # A saída da camada oculta é tanh: derivada local = 1-A².
                delta = (delta @ self.weights[layer].T) * (1.0 - activations[layer] ** 2)
        return grad_weights, grad_biases

    def update(self, grad_weights, grad_biases, learning_rate):
        """Descida do gradiente aplicada a todos os parâmetros."""
        for layer in range(len(self.weights)):
            self.weights[layer] -= learning_rate * grad_weights[layer]
            self.biases[layer] -= learning_rate * grad_biases[layer]

    def fit(self, X, y, learning_rate=0.1, epochs=2000):
        """Treino em lote completo; registra a perda inicial e cada época."""
        history = np.empty(epochs + 1)
        for epoch in range(epochs):
            probabilities, activations, logits = self.forward(X)
            history[epoch], delta = self.loss_and_delta(logits, probabilities, y)
            grad_weights, grad_biases = self.backward(activations, delta)
            self.update(grad_weights, grad_biases, learning_rate)
        probabilities, _, logits = self.forward(X)
        history[-1], _ = self.loss_and_delta(logits, probabilities, y)
        return history

    def predict(self, X):
        probabilities, _, _ = self.forward(X)
        if probabilities.shape[1] == 1:
            return (probabilities[:, 0] >= 0.5).astype(int)
        return np.argmax(probabilities, axis=1)

    @property
    def parameter_count(self):
        return sum(array.size for array in self.weights + self.biases)
