import numpy as np

m = np.random.randint(10, size=(3, 3))

logical = m < 5

indices = np.nonzero(logical)

print(indices)
print(np.transpose(indices))



