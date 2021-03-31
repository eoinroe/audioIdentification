import numpy as np
from matplotlib import pyplot as plt

CQ = np.random.randint(2, size=(3, 3))
print(CQ)

CD = np.random.randint(2, size=(3, 3))
print(CD)

result = CQ == CD
print(result)

percentage = np.sum(result) / CQ.size
print(percentage)

CQ = np.random.randint(2, size=(3, 3))
print('C(Q): \n', CQ)

CD = np.random.randint(2, size=(3, 6))
print('C(D): \n', CD)

rows, cols = np.indices(CQ.shape)

print(CD.shape[1] - CQ.shape[1])

# Shifting the query by m positions yields the constellation map m + C(Q)
m_positions = CD.shape[1] - CQ.shape[1] + 1

# As long as the number of correctly matching
# peak coordinates is statistically significant
percentages = np.zeros(m_positions)

for m in range(m_positions):
    print('m: ', m)

    print('C(Q): \n', CQ)

    shifted = CD[rows, cols + m]
    print('Shifted: \n', shifted)

    result = CQ == shifted

    print('Result: \n', result.astype(int))

    percentage = np.sum(result) / CQ.size
    # print(percentage)

    percentages[m] = percentage

for p in percentages:
    print(p)

print('Maximum: ', np.amax(percentages))

i = np.argmax(percentages)
print('Index of Maximum: ', i)

match = CD[rows, cols + i]

plt.figure(figsize=(10, 5))

y, x = np.nonzero(CD)
plt.scatter(x, y, s=10, marker='.', c='black')

y, x = np.nonzero(match)
plt.scatter(x + i, y, s=1, marker='.', c='red')

plt.show()

# m = np.arange(9).reshape(3, 3)
# i = np.indices((3, 3))
#
# print(m[i])


