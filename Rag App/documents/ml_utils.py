=== FILE: ml_utils.py ===

"""
ml_utils.py
===========
A collection of utility functions and classes for machine learning workflows.
This module covers data preprocessing, model evaluation, feature engineering,
and basic neural network building blocks.

Intended for use in educational RAG system testing and ML pipelines.
"""

import math
import random
from typing import List, Tuple, Dict, Optional, Union


# ─────────────────────────────────────────────
# DATA PREPROCESSING
# ─────────────────────────────────────────────

def normalize(values: List[float]) -> List[float]:
    """
    Normalize a list of values to the range [0, 1] using min-max scaling.

    Min-max normalization transforms each value x using:
        x_norm = (x - x_min) / (x_max - x_min)

    Args:
        values (List[float]): A list of numeric values to normalize.

    Returns:
        List[float]: Normalized values in the range [0, 1].

    Raises:
        ValueError: If all values are identical (zero variance).
    """
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        raise ValueError("Cannot normalize: all values are identical (zero range).")
    return [(v - min_val) / (max_val - min_val) for v in values]


def standardize(values: List[float]) -> List[float]:
    """
    Standardize a list of values to have mean=0 and std=1 (Z-score normalization).

    Formula: z = (x - mean) / std

    Args:
        values (List[float]): Input numeric values.

    Returns:
        List[float]: Standardized values with mean ~0 and std ~1.
    """
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(variance)
    if std == 0:
        raise ValueError("Standard deviation is zero; cannot standardize constant data.")
    return [(v - mean) / std for v in values]


def train_test_split(
    data: List,
    labels: List,
    test_ratio: float = 0.2,
    seed: Optional[int] = None
) -> Tuple[List, List, List, List]:
    """
    Split a dataset into training and testing sets.

    Args:
        data (List): Feature vectors or raw data samples.
        labels (List): Corresponding labels for each data point.
        test_ratio (float): Fraction of data to reserve for testing (default 0.2).
        seed (Optional[int]): Random seed for reproducibility.

    Returns:
        Tuple: (train_data, test_data, train_labels, test_labels)
    """
    if seed is not None:
        random.seed(seed)

    combined = list(zip(data, labels))
    random.shuffle(combined)
    split_idx = int(len(combined) * (1 - test_ratio))

    train = combined[:split_idx]
    test = combined[split_idx:]

    train_data, train_labels = zip(*train) if train else ([], [])
    test_data, test_labels = zip(*test) if test else ([], [])

    return list(train_data), list(test_data), list(train_labels), list(test_labels)


# ─────────────────────────────────────────────
# EVALUATION METRICS
# ─────────────────────────────────────────────

def accuracy(y_true: List[int], y_pred: List[int]) -> float:
    """
    Compute classification accuracy.

    Args:
        y_true (List[int]): Ground truth labels.
        y_pred (List[int]): Predicted labels.

    Returns:
        float: Accuracy score between 0.0 and 1.0.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def precision_recall_f1(
    y_true: List[int], y_pred: List[int], positive_class: int = 1
) -> Dict[str, float]:
    """
    Compute precision, recall, and F1-score for binary classification.

    Definitions:
        Precision = TP / (TP + FP)
        Recall    = TP / (TP + FN)
        F1        = 2 * (Precision * Recall) / (Precision + Recall)

    Args:
        y_true (List[int]): Ground truth binary labels.
        y_pred (List[int]): Predicted binary labels.
        positive_class (int): The label treated as positive (default 1).

    Returns:
        Dict[str, float]: Dictionary with 'precision', 'recall', 'f1' keys.
    """
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive_class and p == positive_class)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != positive_class and p == positive_class)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == positive_class and p != positive_class)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    """
    Compute Mean Squared Error (MSE) for regression tasks.

    MSE = (1/n) * Σ(y_true - y_pred)²

    Args:
        y_true (List[float]): Actual target values.
        y_pred (List[float]): Predicted values.

    Returns:
        float: MSE value (lower is better).
    """
    n = len(y_true)
    return sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / n


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────

def one_hot_encode(labels: List[str]) -> Tuple[List[List[int]], List[str]]:
    """
    One-hot encode a list of categorical string labels.

    Each unique label is assigned an index. Each input label is
    converted to a binary vector with a 1 at its index position.

    Args:
        labels (List[str]): Categorical labels to encode.

    Returns:
        Tuple[List[List[int]], List[str]]:
            - Encoded matrix (list of binary vectors)
            - List of unique classes in sorted order
    """
    classes = sorted(set(labels))
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    encoded = []
    for label in labels:
        vec = [0] * len(classes)
        vec[class_to_idx[label]] = 1
        encoded.append(vec)
    return encoded, classes


def compute_tf_idf(documents: List[str]) -> List[Dict[str, float]]:
    """
    Compute a simplified TF-IDF representation for a list of documents.

    TF  (Term Frequency)  = count(term in doc) / total terms in doc
    IDF (Inverse Document Frequency) = log(N / df(term)), where N = total docs

    Args:
        documents (List[str]): List of raw text documents.

    Returns:
        List[Dict[str, float]]: TF-IDF scores per document as word→score dicts.
    """
    n = len(documents)
    tokenized = [doc.lower().split() for doc in documents]

    # Compute document frequency
    df: Dict[str, int] = {}
    for tokens in tokenized:
        for word in set(tokens):
            df[word] = df.get(word, 0) + 1

    # Compute TF-IDF for each document
    tfidf_docs = []
    for tokens in tokenized:
        tf: Dict[str, float] = {}
        for word in tokens:
            tf[word] = tf.get(word, 0) + 1
        total = len(tokens)
        tfidf = {
            word: (count / total) * math.log(n / df[word])
            for word, count in tf.items()
        }
        tfidf_docs.append(tfidf)

    return tfidf_docs


# ─────────────────────────────────────────────
# SIMPLE NEURAL NETWORK COMPONENTS
# ─────────────────────────────────────────────

class Neuron:
    """
    Represents a single artificial neuron with configurable activation.

    A neuron computes: output = activation(dot(weights, inputs) + bias)

    Attributes:
        weights (List[float]): Synaptic weights for each input.
        bias (float): Bias term added to weighted sum.
        activation (str): Activation function name ('relu', 'sigmoid', 'tanh').
    """

    ACTIVATIONS = ("relu", "sigmoid", "tanh", "linear")

    def __init__(self, n_inputs: int, activation: str = "relu", seed: Optional[int] = None):
        """
        Initialize neuron with random weights.

        Args:
            n_inputs (int): Number of input connections.
            activation (str): Activation function to use.
            seed (Optional[int]): Seed for reproducible weight initialization.
        """
        if activation not in self.ACTIVATIONS:
            raise ValueError(f"Activation must be one of {self.ACTIVATIONS}")
        if seed is not None:
            random.seed(seed)
        self.weights = [random.gauss(0, 0.1) for _ in range(n_inputs)]
        self.bias = 0.0
        self.activation = activation

    def _activate(self, z: float) -> float:
        """Apply the chosen activation function to the pre-activation value z."""
        if self.activation == "relu":
            return max(0.0, z)
        elif self.activation == "sigmoid":
            return 1.0 / (1.0 + math.exp(-z))
        elif self.activation == "tanh":
            return math.tanh(z)
        else:  # linear
            return z

    def forward(self, inputs: List[float]) -> float:
        """
        Perform a forward pass through the neuron.

        Args:
            inputs (List[float]): Input values (must match n_inputs).

        Returns:
            float: Activated output of the neuron.
        """
        if len(inputs) != len(self.weights):
            raise ValueError(f"Expected {len(self.weights)} inputs, got {len(inputs)}.")
        z = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        return self._activate(z)


class DenseLayer:
    """
    A fully connected (dense) layer composed of multiple neurons.

    Each input is connected to every neuron in the layer.

    Attributes:
        neurons (List[Neuron]): Collection of neurons forming the layer.
    """

    def __init__(self, n_inputs: int, n_neurons: int, activation: str = "relu"):
        """
        Initialize the dense layer.

        Args:
            n_inputs (int): Number of input features.
            n_neurons (int): Number of neurons in this layer.
            activation (str): Shared activation function for all neurons.
        """
        self.neurons = [Neuron(n_inputs, activation) for _ in range(n_neurons)]

    def forward(self, inputs: List[float]) -> List[float]:
        """
        Compute the forward pass for all neurons in the layer.

        Args:
            inputs (List[float]): Input feature vector.

        Returns:
            List[float]: Output activations of all neurons.
        """
        return [neuron.forward(inputs) for neuron in self.neurons]


# ─────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Normalize sample data
    raw = [10, 20, 30, 40, 50]
    print("Normalized:", normalize(raw))
    print("Standardized:", standardize(raw))

    # Evaluate dummy predictions
    y_true = [1, 0, 1, 1, 0, 0, 1]
    y_pred = [1, 0, 1, 0, 0, 1, 1]
    print("Accuracy:", accuracy(y_true, y_pred))
    print("Metrics:", precision_recall_f1(y_true, y_pred))

    # One-hot encoding
    labels = ["cat", "dog", "cat", "bird", "dog"]
    encoded, classes = one_hot_encode(labels)
    print("Classes:", classes)
    print("Encoded[0]:", encoded[0])

    # Neural network layer
    layer = DenseLayer(n_inputs=3, n_neurons=4, activation="relu")
    output = layer.forward([0.5, -0.3, 1.2])
    print("Layer output:", output)
