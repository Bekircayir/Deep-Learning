from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import numpy as np

# Daten definieren:
X = np.linspace(0, 10, 100)
y = 2*np.sin(X)+1
X = X[:, np.newaxis]

# Erstelle Multi-layer Perceptron
reg = MLPRegressor(hidden_layer_sizes=(50, 50),
                    activation="relu",
                    solver="adam",
                    learning_rate_init=0.001,
                    max_iter=5000,
                    random_state=1)

# Training
reg.fit(X, y)
print(reg.score(X,y))

# Plothilfe definieren
X_plot = np.linspace(0, 10, 1000)
X_plot = X_plot[:, np.newaxis]
y_plot = reg.predict(X_plot)

# Daten plotten:
plt.figure()
plt.plot(X, y, 'kx', label="data")

# Regressionskurve plotten
plt.plot(X_plot, y_plot, label="approximation")

# Achsenbeschriftung und Legende
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc="lower right")
plt.show()