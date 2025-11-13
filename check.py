import numpy as np

t = np.load('extracted_watermark.npy')

# Basic info
print("Shape:", t.shape)
print("Data type:", t.dtype)
print("Size (total elements):", t.size)

# Statistics
print("\nStatistics:")
print("Min:", t.min())
print("Max:", t.max())
print("Mean:", t.mean())
print("Std Dev:", t.std())

# Display data
print("\nFirst 10 elements:", t.flat[:10])
print("\nArray preview:")
print(t)

# Check for NaN or Inf
print("\nContains NaN:", np.isnan(t).any())
print("Contains Inf:", np.isinf(t).any())
