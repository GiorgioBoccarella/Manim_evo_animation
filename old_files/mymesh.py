import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-4, 4, 200)
y = np.linspace(-4, 4, 200)

sigma, mu = 0.4, [0.0, 0.0]
z = np.exp(-(np.linalg.norm(np.array([x - mu[0], y - mu[1]])) ** 2 / (2.0 * sigma ** 2)))

z = np.array([np.exp(-(np.linalg.norm(np.array([i - mu[0], j - mu[1]])) ** 2 / (2.0 * sigma ** 2))) for j in y for i in x])

X, Y = np.meshgrid(x, y)
Z = z.reshape(200, 200)

plt.pcolor(X, Y, Z)
plt.show()