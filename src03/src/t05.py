from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    img = Image.open("../data/leaves-4673997_1920.jpg")

    # 3*3 blur
    N = 3
    _kernel = np.ones([N,N])/N*N
    kernel = ImageFilter.Kernel(size=(N,N), kernel=_kernel.flatten())
    img_blur9 = img.filter(kernel)
    plt.imshow(img_blur9)
    plt.pause(5)

    # 5*5 blur
    N = 5
    _kernel = np.ones([N,N])/N*N
    kernel = ImageFilter.Kernel(size=(N,N), kernel=_kernel.flatten())
    img_blur25 = img.filter(kernel)
    plt.imshow(img_blur25)
    plt.pause(5)

    # 3*3 blur
    N = 3
    _kernel = np.array([[1, 2, 1],
                        [2, 4, 2],
                        [1, 2, 1]])/16
    kernel = ImageFilter.Kernel(size=(N,N), kernel=_kernel.flatten())
    img_gaussian_blur9 = img.filter(kernel)
    plt.imshow(img_gaussian_blur9)
    plt.pause(5)

    # 5*5 blur
    N = 5
    _kernel = np.array([[1,  4,  6,  4, 1],
                        [4, 16, 24, 16, 4],
                        [6, 24, 36, 24, 6],
                        [4, 16, 24, 16, 4],
                        [1,  4,  6,  4, 1]])/255
    kernel = ImageFilter.Kernel(size=(N,N), kernel=_kernel.flatten())
    img_gaussian_blur25 = img.filter(kernel)
    plt.imshow(img_gaussian_blur25)
    plt.pause(5)

    with open("../data/task5/leaves_blur9.jpg", "w") as f:
        img_blur9.save(f, format="JPEG")

    with open("../data/task5/leaves_blur25.jpg", "w") as f:
        img_blur25.save(f, format="JPEG")

    with open("../data/task5/leaves_gaussian_blur9.jpg", "w") as f:
        img_gaussian_blur9.save(f, format="JPEG")

    with open("../data/task5/leaves_gaussian_blur25.jpg", "w") as f:
        img_gaussian_blur25.save(f, format="JPEG")