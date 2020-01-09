from PIL import Image
import numpy as np

def tone_curve(img: np.ndarray, gamma=1) -> np.ndarray:
    return 255.0 * (img/255.0)**(1/gamma)

if __name__ == "__main__":
    img = Image.open("../data/city-3021474_1920.jpg")

    gamma_converted_array = tone_curve(np.array(img, 'f'), gamma=0.5)
    img_gamma_converted1 = Image.fromarray(np.uint8(gamma_converted_array))

    gamma_converted_array = tone_curve(np.array(img, 'f'), gamma=1.0)
    img_gamma_converted2 = Image.fromarray(np.uint8(gamma_converted_array))

    gamma_converted_array = tone_curve(np.array(img, 'f'), gamma=1.5)
    img_gamma_converted3 = Image.fromarray(np.uint8(gamma_converted_array))

    with open("../data/task1/gamma_0f5.jpg", "w") as f:
        img_gamma_converted1.save(f, format="JPEG")

    with open("../data/task1/gamma_1f0.jpg", "w") as f:
        img_gamma_converted2.save(f, format="JPEG")

    with open("../data/task1/gamma_1f5.jpg", "w") as f:
        img_gamma_converted3.save(f, format="JPEG")

