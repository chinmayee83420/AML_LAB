import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Input X values
x = list(map(float, input("Enter X values : ").split(",")))

# Input Y values
y = list(map(float, input("Enter Y values : ").split(",")))

# Convert to NumPy arrays
X = np.array(x)
Y = np.array(y)

# -----------------------------
# Manual Linear Regression
# -----------------------------

x_mean = np.mean(X)
y_mean = np.mean(Y)

m = np.sum((X - x_mean) * (Y - y_mean)) / np.sum((X - x_mean) ** 2)
c = y_mean - m * x_mean

# Predict Y values
y_pred = m * X + c

# Regression Equation
print("\nRegression Equation:")
print(f"y = {m:.2f}x + {c:.2f}")

# -----------------------------
# Model Metrics
# -----------------------------

mse = mean_squared_error(Y, y_pred)
mae = mean_absolute_error(Y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y, y_pred)

print("\nModel Metrics")
print(f"MSE  : {mse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# -----------------------------
# Graphs
# -----------------------------

plt.figure(figsize=(8, 10))

# Graph 1: Linear Regression
plt.subplot(2, 1, 1)

plt.scatter(X, Y, color="blue", label="Actual Data", s=60)

x_line = np.linspace(min(X), max(X), 100)
y_line = m * x_line + c

plt.plot(x_line, y_line, color="red", linewidth=2, label="Best Fit Line")

plt.title("Linear Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)

# Graph 2: Model Metrics Comparison
plt.subplot(2, 1, 2)

metrics = ["MSE", "MAE", "RMSE", "R²"]
values = [mse, mae, rmse, r2]

plt.scatter(metrics, values, color="purple", s=100)

for i, value in enumerate(values):
    plt.text(i, value + 0.02, f"{value:.4f}",
             ha="center", fontsize=10)

plt.title("Model Metrics Comparison")
plt.xlabel("Evaluation Metrics")
plt.ylabel("Values")
plt.grid(True)

plt.tight_layout()
plt.show()