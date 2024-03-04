import numpy as np

# Lifts x by producing all possible pairwise combinations of coordinates
def lift(x):
    dimension = len(x)
    out_length = dimension + (dimension * (dimension + 1)) // 2
    out = np.ones(out_length)
    
    index = 0
    for val in x:
        out[index] = val
        index += 1
    
    for i in range(dimension):
        for j in range(0, i + 1):
            out[index] = x[i] * x[j]
            index += 1

    return out

# Useful helper function that generates the labels (of x_i * x_j) for a lifted dataset
def liftLabels(dimension):
    out_length = dimension + (dimension * (dimension + 1)) // 2
    labels = [1] * out_length
    
    index = 0
    for val in range(dimension):
        labels[index] = f"x_{val}"
        index += 1
    
    for i in range(dimension):
        for j in range(0, i + 1):
            labels[index] = f"x_{i} * x_{j}"
            index += 1

    return labels

def liftDataset(X):
    return np.array([lift(vector) for vector in X])


