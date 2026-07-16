from skimage.feature import hog
from skimage import color, io
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt


model = VGG16(weights="imagenet",include_top=False)
# Membaca gambar
image = io.imread("bambang.png")

# Konversi ke grayscale
image_gray = color.rgb2gray(image)

# Ekstraksi fitur HOG
features, hog_image = hog(
    image_gray,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    visualize=True
)

# Menampilkan hasil
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.imshow(image_gray, cmap="gray")
ax1.set_title("Gambar Asli")
ax1.axis("off")

ax2.imshow(hog_image, cmap="gray")
ax2.set_title("Ekstraksi Fitur HOG")
ax2.axis("off")

plt.show()

print("Jumlah Fitur:", len(features))
img = image.load_img("bambang.png",target_size=(224,224))
img_array = image.img_to_array(img)
img_array=np.expand_dims(img_array,axis=0)
img_array=preprocess_input(img_array)

features=model.predict(img_array)

print(features)