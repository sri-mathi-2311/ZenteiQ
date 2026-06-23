"""
neural_network.py
------------------
Objective
    Implement a fully-connected (dense) feed-forward neural network
    completely from scratch using only NumPy -- no TensorFlow, PyTorch,
    or Keras. This is the core deliverable of the assignment.

Architecture
    Input(784) -> Dense -> Activation (ReLU or Tanh) -> ... ->
    Dense -> Softmax -> 35-way class probabilities

Implemented from first principles
    - Weight initialization (He init for ReLU, Xavier init for Tanh)
    - Forward propagation (matrix multiplications + activations)
    - Categorical cross-entropy loss (with optional L2 regularization)
    - Backpropagation (manual gradient derivation through every layer)
    - Mini-batch gradient descent parameter update
    - A validation loop that runs after every epoch (no gradient updates)

Design notes
    The network is built as a list of layer sizes, e.g. [784, 128, 64, 35],
    so it generalizes to any number of hidden layers without code changes.
    All math is vectorized with NumPy; there are no Python-level loops over
    individual neurons or samples inside forward/backward passes (only over
    layers, which is necessary for backprop's layer-by-layer chain rule).
"""

import numpy as np


class NeuralNetwork:
    """
    A from-scratch, fully-connected feed-forward neural network trained
    with mini-batch gradient descent and backpropagation.

    Parameters
    ----------
    layer_sizes : list[int]
        Sizes of every layer including input and output, e.g.
        [784, 128, 64, 35] means: 784 input features, two hidden layers
        of 128 and 64 units, and 35 output classes.
    activation : str, default "relu"
        Hidden-layer activation function. One of {"relu", "tanh"}.
    learning_rate : float, default 0.1
        Step size used in the gradient descent parameter update.
    l2_lambda : float, default 0.0
        L2 weight-decay regularization strength. 0.0 disables it.
    seed : int, default 42
        Seed for reproducible weight initialization.
    """

    def __init__(self, layer_sizes, activation="relu", learning_rate=0.1,
                 l2_lambda=0.0, seed=42):
        assert activation in ("relu", "tanh"), "activation must be 'relu' or 'tanh'"
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1  # number of weight matrices
        self.activation_name = activation
        self.lr = learning_rate
        self.l2_lambda = l2_lambda
        self.rng = np.random.default_rng(seed)

        self.weights = []
        self.biases = []
        self._init_weights()

        # Recorded once training starts; useful for plotting / reporting.
        self.history = {"train_loss": [], "train_acc": [],
                         "val_loss": [], "val_acc": []}

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _init_weights(self):
        """
        Initialize weight matrices and bias vectors for every layer.

        Uses He initialization (scale = sqrt(2 / fan_in)) for ReLU
        networks, which keeps activation variance stable across depth,
        and Xavier/Glorot initialization (scale = sqrt(1 / fan_in)) for
        Tanh networks, since Tanh saturates with large pre-activations.
        The final (output) layer always uses Xavier-style scaling because
        it feeds into softmax, not the hidden activation.
        """
        for i in range(self.num_layers):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            is_output_layer = (i == self.num_layers - 1)

            if self.activation_name == "relu" and not is_output_layer:
                scale = np.sqrt(2.0 / fan_in)          # He init
            else:
                scale = np.sqrt(1.0 / fan_in)          # Xavier init

            W = self.rng.normal(0, scale, size=(fan_in, fan_out)).astype(np.float32)
            b = np.zeros((1, fan_out), dtype=np.float32)
            self.weights.append(W)
            self.biases.append(b)

    # ------------------------------------------------------------------
    # Activation functions and their derivatives
    # ------------------------------------------------------------------
    @staticmethod
    def relu(z):
        """ReLU activation: max(0, z). Returns 0 for negative inputs."""
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z):
        """Derivative of ReLU w.r.t. its input z: 1 where z>0, else 0."""
        return (z > 0).astype(z.dtype)

    @staticmethod
    def tanh(z):
        """Tanh activation, squashes inputs to (-1, 1)."""
        return np.tanh(z)

    @staticmethod
    def tanh_derivative(z):
        """Derivative of tanh w.r.t. its input z: 1 - tanh(z)^2."""
        t = np.tanh(z)
        return 1.0 - t ** 2

    @staticmethod
    def softmax(z):
        """
        Numerically-stable softmax over the last axis.

        Subtracting the row-wise max before exponentiating prevents
        overflow for large logits while leaving the result mathematically
        unchanged (softmax is shift-invariant).
        """
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _activate(self, z):
        """Apply the configured hidden-layer activation function."""
        return self.relu(z) if self.activation_name == "relu" else self.tanh(z)

    def _activate_derivative(self, z):
        """Derivative of the configured hidden-layer activation function."""
        return self.relu_derivative(z) if self.activation_name == "relu" else self.tanh_derivative(z)

    # ------------------------------------------------------------------
    # Forward propagation
    # ------------------------------------------------------------------
    def forward(self, X):
        """
        Run a full forward pass through the network.

        Parameters
        ----------
        X : np.ndarray, shape (batch_size, input_dim)

        Returns
        -------
        cache : dict
            Stores every layer's pre-activation (Z) and post-activation
            (A) values. These are needed by backward() to compute
            gradients via the chain rule, and by predict() for inference.
            cache["A0"] is the input itself; cache[f"A{num_layers}"] is
            the final softmax probability output.
        """
        cache = {"A0": X}
        A = X
        for i in range(self.num_layers):
            Z = A @ self.weights[i] + self.biases[i]
            is_output_layer = (i == self.num_layers - 1)
            A = self.softmax(Z) if is_output_layer else self._activate(Z)
            cache[f"Z{i+1}"] = Z
            cache[f"A{i+1}"] = A
        return cache

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def compute_loss(self, y_pred, y_true):
        """
        Categorical cross-entropy loss, with optional L2 regularization.

        Parameters
        ----------
        y_pred : np.ndarray, shape (batch_size, num_classes)
            Softmax probabilities from forward().
        y_true : np.ndarray, shape (batch_size, num_classes)
            One-hot encoded ground-truth labels.

        Returns
        -------
        float
            Mean cross-entropy loss over the batch, plus the L2 penalty
            term (sum of squared weights, scaled by l2_lambda) if enabled.
        """
        eps = 1e-9  # avoids log(0)
        n = y_true.shape[0]
        data_loss = -np.sum(y_true * np.log(y_pred + eps)) / n

        if self.l2_lambda > 0:
            l2_term = sum(np.sum(W ** 2) for W in self.weights)
            data_loss += (self.l2_lambda / (2 * n)) * l2_term
        return data_loss

    # ------------------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------------------
    def backward(self, cache, y_true):
        """
        Compute gradients of the loss w.r.t. every weight and bias using
        backpropagation (the chain rule applied layer by layer, from the
        output back to the input).

        Key derivative used:
            For softmax + categorical cross-entropy combined, the
            gradient of the loss w.r.t. the OUTPUT layer's pre-activation
            Z simplifies elegantly to (y_pred - y_true). This is a
            standard and important simplification that avoids having to
            separately differentiate softmax and cross-entropy.

        Parameters
        ----------
        cache : dict
            The dictionary returned by forward(), containing every
            layer's Z and A values.
        y_true : np.ndarray, shape (batch_size, num_classes)
            One-hot ground-truth labels.

        Returns
        -------
        grads_W, grads_b : list[np.ndarray], list[np.ndarray]
            Gradients for every weight matrix / bias vector, ordered the
            same way as self.weights / self.biases.
        """
        n = y_true.shape[0]
        grads_W = [None] * self.num_layers
        grads_b = [None] * self.num_layers

        # Output layer: dL/dZ_out = y_pred - y_true  (softmax + CE shortcut)
        A_out = cache[f"A{self.num_layers}"]
        dZ = A_out - y_true

        for i in reversed(range(self.num_layers)):
            A_prev = cache[f"A{i}"]  # activation feeding INTO layer i
            grads_W[i] = (A_prev.T @ dZ) / n
            grads_b[i] = np.sum(dZ, axis=0, keepdims=True) / n

            if self.l2_lambda > 0:
                grads_W[i] += (self.l2_lambda / n) * self.weights[i]

            if i > 0:
                # Propagate error to the previous layer through W_i,
                # then through that layer's own activation derivative.
                dA_prev = dZ @ self.weights[i].T
                dZ = dA_prev * self._activate_derivative(cache[f"Z{i}"])

        return grads_W, grads_b

    # ------------------------------------------------------------------
    # Gradient descent parameter update
    # ------------------------------------------------------------------
    def update_params(self, grads_W, grads_b):
        """
        Apply one step of (mini-batch) gradient descent:
            W <- W - learning_rate * dW
            b <- b - learning_rate * db
        """
        for i in range(self.num_layers):
            self.weights[i] -= self.lr * grads_W[i]
            self.biases[i] -= self.lr * grads_b[i]

    # ------------------------------------------------------------------
    # Prediction / evaluation helpers
    # ------------------------------------------------------------------
    def predict_proba(self, X):
        """Return softmax class probabilities for input X."""
        return self.forward(X)[f"A{self.num_layers}"]

    def predict(self, X):
        """Return the predicted integer class index for each row of X."""
        return np.argmax(self.predict_proba(X), axis=1)

    def accuracy(self, X, y_true_int):
        """
        Compute classification accuracy.

        Parameters
        ----------
        X : np.ndarray, shape (n, input_dim)
        y_true_int : np.ndarray, shape (n,)
            Integer (not one-hot) ground-truth labels.
        """
        preds = self.predict(X)
        return float(np.mean(preds == y_true_int))

    # ------------------------------------------------------------------
    # Training loop (mini-batch gradient descent + validation)
    # ------------------------------------------------------------------
    def train(self, X_train, y_train_oh, y_train_int,
               X_val, y_val_oh, y_val_int,
               epochs=60, batch_size=64, verbose=True):
        """
        Train the network with mini-batch gradient descent, running a
        validation pass (forward only, no parameter updates) after every
        epoch.

        Order followed each mini-batch, exactly as specified by the
        assignment: forward propagation -> loss -> backpropagation ->
        gradient descent update.

        Parameters
        ----------
        X_train, y_train_oh, y_train_int : training inputs, one-hot
            labels, and integer labels.
        X_val, y_val_oh, y_val_int : validation inputs, one-hot labels,
            and integer labels. Used ONLY for monitoring; gradients are
            never computed on this set.
        epochs : int
            Number of full passes over the training data.
        batch_size : int
            Mini-batch size for gradient descent.
        verbose : bool
            If True, print loss/accuracy after every epoch.

        Returns
        -------
        self.history : dict
            train/val loss and accuracy recorded per epoch.
        """
        n = X_train.shape[0]
        for epoch in range(1, epochs + 1):
            # Shuffle every epoch so mini-batches differ each pass.
            perm = self.rng.permutation(n)
            X_shuf, y_shuf = X_train[perm], y_train_oh[perm]

            for start in range(0, n, batch_size):
                end = start + batch_size
                X_batch = X_shuf[start:end]
                y_batch = y_shuf[start:end]

                cache = self.forward(X_batch)                       # 1. forward propagation
                grads_W, grads_b = self.backward(cache, y_batch)     # 2. backpropagation
                self.update_params(grads_W, grads_b)                 # 3. gradient descent

            # ---- End-of-epoch metrics (train + validation) ----
            train_pred_proba = self.predict_proba(X_train)
            train_loss = self.compute_loss(train_pred_proba, y_train_oh)
            train_acc = float(np.mean(np.argmax(train_pred_proba, axis=1) == y_train_int))

            val_pred_proba = self.predict_proba(X_val)               # 4. validation loop (no updates)
            val_loss = self.compute_loss(val_pred_proba, y_val_oh)
            val_acc = float(np.mean(np.argmax(val_pred_proba, axis=1) == y_val_int))

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == epochs):
                print(f"Epoch {epoch:3d}/{epochs} | "
                      f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                      f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        return self.history

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path):
        """Save all weights and biases to a .npz file."""
        save_dict = {}
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            save_dict[f"W{i}"] = W
            save_dict[f"b{i}"] = b
        save_dict["layer_sizes"] = np.array(self.layer_sizes)
        save_dict["activation"] = np.array(self.activation_name)
        np.savez_compressed(path, **save_dict)

    def load(self, path):
        """Load weights and biases previously saved with save()."""
        data = np.load(path, allow_pickle=True)
        self.weights = [data[f"W{i}"] for i in range(self.num_layers)]
        self.biases = [data[f"b{i}"] for i in range(self.num_layers)]
