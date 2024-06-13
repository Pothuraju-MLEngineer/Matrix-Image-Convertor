import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from ex_matrices import image_1, image_2, image_3, image_4, image_5  # Import the image matrix

# Define the kernels
horizontal_kernel = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
])

vertical_kernel = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
])

# Perform convolution
convolved_matrix_horizontal = convolve2d(image_5, horizontal_kernel, mode='valid')
convolved_matrix_vertical = convolve2d(image_5, vertical_kernel, mode='valid')

# Save the resulting matrices to a file
with open('results.txt', 'w') as f:
    f.write("Convolved Matrix - Horizontal Edges:\n")
    np.savetxt(f, convolved_matrix_horizontal, fmt='%d')
    f.write("\nConvolved Matrix - Vertical Edges:\n")
    np.savetxt(f, convolved_matrix_vertical, fmt='%d')

# Plot the original and the convolved images
fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # Adjusted for better visibility

# Original image
axs[0].imshow(image_5, cmap='gray', interpolation='none')
axs[0].set_title("Original Image")
axs[0].set_xlabel("Pixel X")  # X-axis label
axs[0].set_ylabel("Pixel Y")  # Y-axis label
axs[0].axis('on')  # Show axes

# Convolved image - Horizontal edges
axs[1].imshow(convolved_matrix_horizontal, cmap='gray', interpolation='none')
axs[1].set_title("Detected Horizontal Edges")
axs[1].set_xlabel("Pixel X")
axs[1].set_ylabel("Pixel Y")
axs[1].axis('on')

# Convolved image - Vertical edges
axs[2].imshow(convolved_matrix_vertical, cmap='gray', interpolation='none')
axs[2].set_title("Detected Vertical Edges")
axs[2].set_xlabel("Pixel X")
axs[2].set_ylabel("Pixel Y")
axs[2].axis('on')

plt.show()  # Display the figure
