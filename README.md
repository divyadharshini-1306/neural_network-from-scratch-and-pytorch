# Neural Network From Scratch and in PyTorch

A small neural network implemented two ways:

1. **From scratch**, using only NumPy — every line of forward propagation,
   backpropagation, and gradient descent written and explained explicitly.
2. **In PyTorch**, using `nn.Module` and autograd — the same architecture,
   to show the underlying math is the same thing the framework automates.

The goal of this project is to demonstrate a working understanding of the
math behind neural networks (linear algebra, calculus / chain rule,
gradient-based optimization) — not just the ability to call a library.

## Why two versions?

Anyone can train a model with PyTorch. Showing that a hand-written
implementation **converges to the same result** as PyTorch's autograd is
stronger evidence of understanding: it proves the from-scratch backprop
math is correct, not just plausible-looking.

<img width="975" height="582" alt="image" src="https://github.com/user-attachments/assets/ddcfe8b9-5d88-4be4-bd07-abae0c840bd4" />


Both versions are trained on the same XOR problem, same architecture
(2 inputs → 4 hidden neurons → 1 output), same learning rate. Both reach
a final loss below 0.002, confirming the from-scratch implementation and
PyTorch's autograd are computing the same underlying math.

## The problem: XOR

XOR is the classic example used to demonstrate why neural networks need a
**hidden layer**. A single layer (no hidden layer) can only separate two
classes with a straight line — and there is no straight line that
correctly separates XOR's four points:

| Input  | Output |
|--------|--------|
| (0, 0) | 0      |
| (0, 1) | 1      |
| (1, 0) | 1      |
| (1, 1) | 0      |

Plot these four points: the two `1`s and two `0`s sit on alternating
corners of a square. Any straight line leaves at least one point
misclassified. A hidden layer lets the network bend that line into a
curve — this is precisely the historical example (Minsky & Papert, 1969)
that motivated multi-layer networks.

## The math, briefly

**Forward pass** (per layer):
```
z = W @ x + b        (weighted sum of inputs, plus bias)
a = activation(z)    (non-linearity: sigmoid or ReLU)
```

**Loss** (Mean Squared Error):
```
loss = mean((prediction - target)^2)
```

**Backward pass** (chain rule, applied per layer, output → input):
```
grad_z = grad_a * activation_derivative(z)   # undo the activation
grad_W = grad_z @ x.T                         # gradient w.r.t. this layer's weights
grad_x = W.T @ grad_z                         # gradient passed to the PREVIOUS layer
```

**Gradient descent update:**
```
W -= learning_rate * grad_W
b -= learning_rate * grad_b
```

`grad_x` is the key line — it's what makes this *back*propagation: each
layer computes a gradient for the layer before it, daisy-chaining
backward through the whole network.

## Project structure

```
.
├── from_scratch/
│   ├── neural_network.py     # Layer + Network classes, manual backprop
│   └── train_xor.py          # trains the from-scratch network on XOR
├── pytorch_version/
│   ├── pytorch_network.py     # same architecture, built with nn.Module
│   └── train_xor_torch.py     # trains the PyTorch network on XOR
├── compare_losses.py          # runs both versions, plots loss curves together
├── decision_boundary.py       # trains on a richer 2D dataset, visualizes the learned boundary
├── outputs/                    # generated plots (created when scripts are run)
└── requirements.txt
```

## Running it

### Locally
```bash
pip install -r requirements.txt

cd from_scratch
python3 train_xor.py

cd ../pytorch_version
python3 train_xor_torch.py

cd ..
python3 compare_losses.py
python3 decision_boundary.py
```

### In Google Colab
Recreate the folder structure with `os.makedirs(...)`, write each `.py`
file into place using `%%writefile`, then run each script with `!python`,
`cd`-ing into the right folder first using `%cd` (each script saves its
output image using a relative path, so the working directory matters).
View generated images inline with `IPython.display.Image`.

## From-scratch vs. PyTorch: what's actually different

| Concept                  | From scratch (`neural_network.py`)        | PyTorch (`pytorch_network.py`)         |
|---------------------------|--------------------------------------------|------------------------------------------|
| Weights & biases           | `self.W = np.random.randn(...)`             | `nn.Linear(n_in, n_out)`                 |
| Forward pass                | `self.W @ x + self.b`                        | `self.linear1(x)`                        |
| Tracking values for backprop | Manually cached as `self.x`, `self.z`          | Automatic — PyTorch's **autograd** builds a computation graph during forward() |
| Backward pass                 | Manually coded chain rule (`grad_z`, `grad_W`, `grad_x`) | `loss.backward()`                        |
| Weight update                   | `self.W -= learning_rate * grad_W`               | `optimizer.step()`                       |
| Gradient reset                   | Not needed (`grad_W` recomputed fresh each call)  | `optimizer.zero_grad()` — **required**, since PyTorch *accumulates* gradients by default |

## A real failure mode worth noting (not just a feature)

`decision_boundary.py` originally used `sigmoid` for the hidden layer (to
match the XOR network). On the harder **moons** dataset, training stalled
at a loss of ~0.09 with a perfectly **straight** decision boundary — a
textbook **vanishing gradient** problem: sigmoid's derivative is at most
0.25 and shrinks fast away from zero, so gradients flowing backward
through it become too small to keep improving the network.

Switching the hidden layer to **ReLU** (`max(0, z)`, derivative = 1 for
any positive input) fixed this immediately — loss kept decreasing well
past 0.09, and the boundary bent to follow the moon shapes:
<img width="647" height="492" alt="image" src="https://github.com/user-attachments/assets/e2b9e84d-e05a-447c-8140-29b014f7dbc6" />


The piecewise-linear "kinked" look of the boundary is itself a direct
consequence of using ReLU, which is piecewise-linear — a sigmoid/tanh
network would produce a smoother, rounded boundary instead.

## What this project demonstrates

- **Linear algebra**: weight matrices, dot products, matrix-vector multiplication
- **Calculus**: derivatives of activation functions, the chain rule applied
  layer by layer (backpropagation)
- **Optimization**: gradient descent, learning rate behavior, the vanishing
  gradient problem and why ReLU addresses it
- **Automatic differentiation**: translating hand-derived gradients into
  PyTorch's autograd, and verifying the two agree
- **Engineering practice**: reproducible results (fixed random seeds),
  side-by-side validation of a custom implementation against an
  industry-standard framework

## Requirements

```
numpy>=1.26
matplotlib>=3.8
torch>=2.0
scikit-learn>=1.3
```
