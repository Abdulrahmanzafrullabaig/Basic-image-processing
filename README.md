### **Image processing**


**Image processing** is a technique used to perform operations on digital images to enhance or extract useful information from them. It involves the use of algorithms to transform, analyze, and manipulate images in various ways. Image processing can be broadly divided into the following stages:

### 1. **Image Acquisition:**
   - This is the first step, where an image is captured using a device like a camera or scanner.

### 2. **Preprocessing:**
   - Techniques such as noise removal, resizing, and normalization are applied to prepare the image for further analysis. This ensures better accuracy in the next steps.

### 3. **Enhancement:**
   - Adjusting brightness, contrast, and sharpness to improve the visibility of features in the image. Filters may also be applied to smooth or sharpen the image.

### 4. **Transformation:**
   - This involves altering the image for specific tasks like rotation, scaling, or applying geometric transformations.

### 5. **Segmentation:**
   - Dividing the image into meaningful regions, such as separating objects from the background, is often key for further analysis.

### 6. **Feature Extraction:**
   - Detecting specific characteristics in the image, such as edges, corners, textures, or patterns that can be used for identifying objects or making decisions.

### 7. **Analysis and Interpretation:**
   - Based on the extracted features, algorithms can classify objects, recognize patterns, or provide measurements for use in various applications, like medical diagnosis, object recognition, or facial detection.

### 8. **Output:**
   - The final processed image is the result, which can be used for visualization, interpretation, or further data processing.

### Applications of Image Processing:
- **Medical Imaging:** Enhancing images from MRI, X-rays, or CT scans for better diagnosis.
- **Computer Vision:** Used in facial recognition, self-driving cars, and robotics for object detection.
- **Satellite Imaging:** Processing images captured from space for terrain analysis or weather forecasting.
- **Photography:** Applying filters, adjusting contrast, and removing noise in photos.
- **Augmented Reality:** Enhancing live camera feeds with virtual information or objects.

Image processing techniques often involve mathematical operations and can be carried out through software like Python libraries (OpenCV, Pillow), MATLAB, or specialized image processing tools.

### ***Thing that we learn from this activty***
Here are the topics formatted as links:

1. [Finding mean, area, min and max of an image](#)
2. [Adding filters to an image](#)
3. [Convert an image to grayscale](#)
4. [Create a histogram](#)
5. [Create an RGB histogram](#)
6. [Add different augmented images](#)
7. [Add different contrast to an image](#)
8. [Add noise and remove noise from an image](#)


---

### 1. [Finding Mean, Area, Min, and Max of an Image](#)

- **Mean:** The average pixel value across the image. This helps to understand the brightness of the image.
- **Area:** For a grayscale image, the area is typically the number of pixels, calculated as `width × height`. For a color image, the total pixel count is the same, but it includes RGB channels.
- **Min/Max:** The minimum and maximum pixel values in the image, which define the darkest and brightest pixels.

**Code Example:**

```python
from PIL import Image
import numpy as np

# Load the image
image = Image.open('/Users/My Computer/Downloads/Basic image porcessing/1.jpg').convert('L')  # Convert to grayscale

# Convert the image to a NumPy array
image_array = np.array(image)

# Calculate the area (number of pixels)
area = image_array.size

# Calculate the mean, min, and max pixel values
mean_val = np.mean(image_array)
min_val = np.min(image_array)
max_val = np.max(image_array)

print(f"Area: {area} pixels")
print(f"Mean pixel value: {mean_val}")
print(f"Minimum pixel value: {min_val}")
print(f"Maximum pixel value:{max_val}")
```
![Screenshot_8-10-2024_03047_localhost](https://github.com/user-attachments/assets/8a126d66-f168-48b9-85f6-e4308e920777)

---

### 2. [Adding Filters to an Image](#)

Filters are applied to an image to emphasize certain features or reduce noise. Common filters include **blur**, **sharpen**, and **edge detection**.

**Code Example (applying a blur filter):**

```python
from PIL import Image
import matplotlib.pyplot as plt

# Load the image
image = Image.open('/Users/My Computer/Downloads/Basic image porcessing/1.jpg')

# Resize the image to specific dimensions
resized_image = image.resize((800, 600))  # Resize to 800x600 pixels

# Scale the image by a factor (e.g., 50%)
scale_factor = 0.5
scaled_image = image.resize((int(image.width * scale_factor), int(image.height * scale_factor)))

# Display the original, resized, and scaled images
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(resized_image)
axes[1].set_title('Resized Image (800x600)')
axes[1].axis('off')

axes[2].imshow(scaled_image)
axes[2].set_title('Scaled Image (50%)')
axes[2].axis('off')

plt.show()

```
![Screenshot_8-10-2024_0326_localhost](https://github.com/user-attachments/assets/ed7f9571-1252-469f-8bac-8fda3f4d80e0)
![Screenshot_8-10-2024_03227_localhost](https://github.com/user-attachments/assets/c4c93e82-e326-4f75-8648-08052c5b30bf)

---

### 3. [Convert an Image to Grayscale](#)

Converting an image to grayscale removes the color information and represents the image using shades of gray, which is useful in reducing complexity for analysis.

**Code Example:**

```python
from PIL import Image
import matplotlib.pyplot as plt

# Load the image
image = Image.open('/Users/My Computer/Downloads/Basic image porcessing/1.jpg')

# Convert the image to grayscale
grayscale_image = image.convert('L')

# Save the grayscale image
grayscale_image.save('grayscale_image.jpg')

# Display the original and grayscale images
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(grayscale_image, cmap='gray')
axes[1].set_title('Grayscale Image')
axes[1].axis('off')

plt.show()


```

---

### 4. [Create a Histogram](#)

A histogram displays the distribution of pixel intensities in an image. This helps to understand the contrast and brightness spread.

**Code Example:**

```python
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

# Load the image
image = Image.open('/Users/My Computer/Downloads/Basic image porcessing/1.jpg')

# Apply different filters
blurred_image = image.filter(ImageFilter.BLUR)
contour_image = image.filter(ImageFilter.CONTOUR)
detail_image = image.filter(ImageFilter.DETAIL)
edge_enhance_image = image.filter(ImageFilter.EDGE_ENHANCE)
emboss_image = image.filter(ImageFilter.EMBOSS)
sharpen_image = image.filter(ImageFilter.SHARPEN)

# Display the original and filtered images
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes[0, 0].imshow(image)
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(blurred_image)
axes[0, 1].set_title('Blurred Image')
axes[0, 1].axis('off')

axes[0, 2].imshow(contour_image)
axes[0, 2].set_title('Contour Image')
axes[0, 2].axis('off')

axes[1, 0].imshow(detail_image)
axes[1, 0].set_title('Detail Image')
axes[1, 0].axis('off')

axes[1, 1].imshow(edge_enhance_image)
axes[1, 1].set_title('Edge Enhance Image')
axes[1, 1].axis('off')

axes[1, 2].imshow(emboss_image)
axes[1, 2].set_title('Emboss Image')
axes[1, 2].axis('off')

plt.show()

```

---

### 5. [Create an RGB Histogram](#)

An RGB histogram shows the distribution of red, green, and blue pixel values in a color image. Each channel is plotted separately.

**Code Example:**

```python
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load the image
image_path = '/Users/My Computer/Downloads/Basic image porcessing/1.jpg'
image = Image.open(image_path)

# Split the image into its respective RGB channels
r, g, b = image.split()

# Convert channels to numpy arrays
r_array = np.array(r).flatten()
g_array = np.array(g).flatten()
b_array = np.array(b).flatten()

# Plot the histograms
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.hist(r_array, bins=256, color='red', alpha=0.5)
plt.title('Red Channel Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.subplot(3, 1, 2)
plt.hist(g_array, bins=256, color='green', alpha=0.5)
plt.title('Green Channel Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.subplot(3, 1, 3)
plt.hist(b_array, bins=256, color='blue', alpha=0.5)
plt.title('Blue Channel Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

```

---

### 6. [Add Different Augmented Images](#)

Image augmentation involves transformations like flipping, rotating, or scaling to create variations of an image, which is useful for training machine learning models.

**Code Example (rotation and flipping):**

```python
from PIL import Image, ImageEnhance, ImageOps
import matplotlib.pyplot as plt
import numpy as np

# Function to apply image augmentations and generate multiple augmented images
def augment_images(image, num_augmented_images):
    augmented_images = []
    for _ in range(num_augmented_images):
        # Randomly apply transformations
        img = image.copy()
        if np.random.rand() > 0.5:
            img = ImageOps.mirror(img)
        if np.random.rand() > 0.5:
            img = img.rotate(np.random.uniform(-40, 40))
        if np.random.rand() > 0.5:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(np.random.uniform(0.5, 1.5))
        if np.random.rand() > 0.5:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(np.random.uniform(0.5, 1.5))
        augmented_images.append(img)
    return augmented_images

# Load an image using Pillow
image_path = '/Users/My Computer/Downloads/Basic image porcessing/1.jpg'
image = Image.open(image_path)

# Generate 20 augmented images
num_augmented_images = 20
augmented_images = augment_images(image, num_augmented_images)

# Display the augmented images
plt.figure(figsize=(15, 15))
for i in range(num_augmented_images):
    plt.subplot(5, 4, i + 1)  # 5 rows, 4 columns layout
    plt.imshow(augmented_images[i])
    plt.title(f'Augmented Image {i+1}')
    plt.axis('off')

plt.tight_layout()
plt.show()

```

---

### 7. [Add Different Contrast to an Image](#)

Adjusting contrast changes the difference between the brightest and darkest parts of the image, which can enhance visibility.

**Code Example (using ImageEnhance):**

```python
 from PIL import Image, ImageEnhance, ImageOps
import matplotlib.pyplot as plt
import numpy as np

# Load the image using Pillow
image_path = '/Users/My Computer/Downloads/Basic image porcessing/1.jpg'
image = Image.open(image_path)

# Convert the image to grayscale
gray_image = ImageOps.grayscale(image)

# 1. Contrast Enhancement using Histogram Equalization
def enhance_contrast_histogram_equalization(gray_image):
    hist_eq_image = ImageOps.equalize(gray_image)
    return hist_eq_image

# 2. Contrast Enhancement using CLAHE (simulated with Pillow)
def enhance_contrast_clahe(gray_image, clip_limit=2.0, tile_grid_size=(8, 8)):
    # Pillow does not have CLAHE, so we simulate it with a simple contrast enhancement
    enhancer = ImageEnhance.Contrast(gray_image)
    clahe_image = enhancer.enhance(clip_limit)
    return clahe_image

# 3. Manual Contrast Adjustment (linear scaling)
def adjust_contrast_manually(image, alpha=1.5, beta=0):
    # Alpha: Contrast control (1.0-3.0), Beta: Brightness control (0-100)
    enhancer = ImageEnhance.Contrast(image)
    adjusted_image = enhancer.enhance(alpha)
    enhancer = ImageEnhance.Brightness(adjusted_image)
    adjusted_image = enhancer.enhance(beta)
    return adjusted_image

# Perform contrast enhancement
hist_eq_image = enhance_contrast_histogram_equalization(gray_image)
clahe_image = enhance_contrast_clahe(gray_image)
manually_adjusted_image = adjust_contrast_manually(image, alpha=2.0, beta=1.5)  # Increasing contrast and brightness

# Display the original and contrast-enhanced images
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(hist_eq_image, cmap='gray')
plt.title('Histogram Equalization')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(clahe_image, cmap='gray')
plt.title('CLAHE')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(manually_adjusted_image)
plt.title('Manual Contrast Adjustment')
plt.axis('off')

plt.tight_layout()
plt.show()

```

---

### 8. [Add Noise and Remove Noise from an Image](#)

Noise can be added to simulate real-world conditions or removed to improve image quality. Adding noise makes the image grainy, while denoising removes unwanted variations.

**Code Example (Adding noise):**

```python
from PIL import Image, ImageFilter, ImageOps
import matplotlib.pyplot as plt
import numpy as np

# Function to add salt and pepper noise to an image
def add_salt_and_pepper_noise(image, amount=0.05):
    np_image = np.array(image)
    row, col = np_image.shape
    num_salt = np.ceil(amount * row * col)
    num_pepper = np.ceil(amount * row * col)

    # Add salt noise
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in np_image.shape]
    np_image[coords[0], coords[1]] = 255

    # Add pepper noise
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in np_image.shape]
    np_image[coords[0], coords[1]] = 0

    return Image.fromarray(np_image)

# Function to remove noise using median filter
def remove_noise(image):
    return image.filter(ImageFilter.MedianFilter(size=3))

# Load the image
image_path = '/Users/My Computer/Downloads/Basic image porcessing/1.jpg'
image = Image.open(image_path).convert('L')  # Convert to grayscale

# Add noise to the image
noisy_image = add_salt_and_pepper_noise(image)

# Remove noise from the image
denoised_image = remove_noise(noisy_image)

# Display the original, noisy, and denoised images
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
titles = ['Original Image', 'Noisy Image', 'Denoised Image']

for ax, img, title in zip(axes, [image, noisy_image, denoised_image], titles):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show()

```

---

