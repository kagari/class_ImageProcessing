from PIL import Image
import numpy as np

if __name__ == "__main__":
    im1 = Image.open("../data/emotions-4637323_1920.jpg")
    im2 = Image.open("../data/anise-2785512_1920.jpg")
    
    mask = np.tile(np.linspace(0, 255, 1920), (1280, 1))
    mask = Image.fromarray(np.uint8(mask))
    
    im_alpha_blending = Image.composite(im1, im2, mask)
    
    m_alpha_blending.save('../data/task2/alpha_blending.jpg', quality=300)