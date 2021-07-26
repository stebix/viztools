import numpy as np
import matplotlib.pyplot as plt

from vizfuncs import create_alpha_cmap

label_data = []

for i in range(1, 5):
    label_data.append(
        np.random.default_rng().integers(low=0, high=i, size=(10, 10))
    )

label_data = np.stack(label_data, axis=0)

assert label_data.shape == (4, 10, 10)

# At first index position 0, the 2D array is 0 everwhere
# Use tis to investigate

cmap = create_alpha_cmap((0, 0, 1), max_alpha=0.333)

fig, ax = plt.subplots()

mppl = ax.imshow(label_data[3, ...], cmap=cmap)

print(mppl.get_cmap())

plt.show()