import numpy as np
y = np.load("y_transformer.npy")
print("Labels únicos:", np.unique(y))
print("Número de classes:", len(np.unique(y)))
print("Máximo label:", np.max(y))