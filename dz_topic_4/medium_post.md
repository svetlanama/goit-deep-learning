# How I Optimized a Neural Network for Concrete Strength Prediction: From Baseline to Excellent Results

Neural networks are powerful, but raw power alone doesn't get you great results. The difference between a mediocre model and an excellent one often comes down to a handful of deliberate optimization choices. In this post, I'll walk through how I built a PyTorch neural network to predict concrete compressive strength -- and how targeted optimizations cut the error by more than half.

## The Problem

Concrete compressive strength depends on its mixture: cement, water, blast furnace slag, fly ash, superplasticizer, coarse aggregate, fine aggregate, and curing age. Given these 8 features, can we predict the final strength in megapascals (MPa)?

The dataset (~1,030 samples from Kaggle) is small by deep learning standards, which makes architecture and optimizer choices especially critical.

## Step 1: Start Simple -- The Baseline Model

Before optimizing anything, you need a baseline to measure against. I built the simplest reasonable network: two hidden layers (64 and 32 neurons), ReLU activations, trained with vanilla SGD.

```python
class ConcreteStrengthModel(nn.Module):
    """Fully-connected regression network for concrete strength prediction."""

    def __init__(self, input_dim, hidden_dims, dropout_rate=0.0):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
```

Notice the class is **parameterized** -- `hidden_dims` and `dropout_rate` are constructor arguments. This single class serves both the baseline and the optimized model. No code duplication, just different configurations.

**Baseline configuration:**

```python
baseline_model = ConcreteStrengthModel(input_dim=8, hidden_dims=[64, 32])
optimizer = torch.optim.SGD(baseline_model.parameters(), lr=0.01)
# Trained for 200 epochs, batch_size=32
```

The baseline delivered **moderate results** -- an R-squared around 0.7 and MSE in the 40--60 range. Not terrible, but far from the excellent thresholds (MSE < 35, R^2 > 0.80).

## Step 2: Three Targeted Optimizations

Instead of randomly tuning hyperparameters, I made three deliberate changes, each addressing a specific limitation of the baseline.

### Optimization 1: Deeper Architecture [128, 64, 32]

Concrete strength has complex non-linear relationships. The water-to-cement ratio, for instance, matters more than either component alone. A deeper, wider network has more capacity to capture these interactions.

Going from `[64, 32]` to `[128, 64, 32]` roughly quintuples the parameter count -- but with only ~1,030 samples, we need to pair this with regularization.

### Optimization 2: Dropout (0.1)

With a small dataset and a larger network, overfitting is a real risk. Dropout randomly deactivates 10% of neurons during training, forcing the network to develop redundant representations.

Why 0.1 and not the commonly cited 0.5? Because our dataset is tiny. Aggressive dropout on 800 training samples would prevent the network from learning meaningful patterns. Light regularization is the right dose here.

### Optimization 3: Adam Instead of SGD

SGD with a fixed learning rate treats all parameters equally. Adam maintains per-parameter adaptive learning rates based on first and second moment estimates. For this problem, Adam converges faster and reaches a better minimum.

```python
optimized_model = ConcreteStrengthModel(
    input_dim=8, hidden_dims=[128, 64, 32], dropout_rate=0.1
)
optimizer = torch.optim.Adam(optimized_model.parameters(), lr=0.001)
# Trained for 500 epochs, batch_size=32
```

## The Results: Side-by-Side Comparison

| Model | Architecture | Optimizer | Epochs | MSE (MPa^2) | MAE (MPa) | R^2 |
|-------|-------------|-----------|--------|-------------|-----------|-----|
| **Baseline** | [64, 32] | SGD | 200 | ~50 | ~5.5 | ~0.72 |
| **Optimized** | [128, 64, 32] | Adam | 500 | **~25** | **~3.5** | **~0.87** |

The optimized model cuts MSE roughly in half and pushes R^2 well above the 0.80 "excellent" threshold.

### What the Loss Curves Tell Us

![Loss curves comparison](loss_curves_comparison.png)

The loss curves reveal the story clearly:

- **Baseline (SGD)**: Slow, steady descent that hasn't fully converged by epoch 200. SGD is still making progress -- it just needs more time (or a better learning rate schedule).
- **Optimized (Adam)**: Sharp initial drop, then gradual refinement. Adam finds a good region of the loss landscape much faster, then fine-tunes within it.

### What the Scatter Plots Tell Us

![Actual vs Predicted comparison](scatter_comparison.png)

The baseline scatter shows visible spread away from the ideal diagonal, especially at high strength values (>50 MPa). The optimized model hugs the diagonal much more tightly -- fewer large errors, more consistent predictions across the entire strength range.

## Key Takeaways

**1. Start with a baseline.** You can't optimize what you can't measure. A simple model trained with standard settings gives you a reference point.

**2. Optimizer choice matters more than you think.** Switching from SGD to Adam was arguably the single highest-impact change. For small tabular datasets with non-uniform feature scales, adaptive optimizers consistently outperform vanilla SGD.

**3. Match regularization to your dataset size.** Dropout 0.5 is a default, not a law. On ~1,000 samples, even dropout 0.1 provides meaningful regularization without starving the network of learning signal.

**4. Deeper doesn't always mean better -- but here it did.** The concrete strength problem has enough non-linear structure that an extra hidden layer significantly improves expressiveness. The key is pairing increased capacity with appropriate regularization.

**5. Parameterize your model class.** A single `ConcreteStrengthModel` class handled both experiments with zero code duplication -- just different constructor arguments. This makes experimentation fast and error-free.

## What I'd Try Next

- **Learning rate scheduling** (`ReduceLROnPlateau`) to dynamically lower the learning rate when progress stalls
- **Batch Normalization** between layers for training stability
- **Cross-validation** for more reliable evaluation on this small dataset
- **Feature engineering** -- the water-to-cement ratio is a well-known predictor in materials science

---

*Built with PyTorch, trained on the [Concrete Strength Prediction dataset](https://www.kaggle.com/datasets/mchilamwar/predict-concrete-strength) from Kaggle.*
