import sys
import numpy as np
from skimage import filters
from PIL import Image, ImageFilter, ImageOps

def _gamma(img: Image, r: float=1.0) -> Image:
    """
    ガンマ変換を行う
    """
    _tmp = 255*(np.array(img.convert('L'))/255)**(1/r)
    img_gamma = Image.fromarray(_tmp.astype(np.uint8))
    return img_gamma

def _edge_detection(img: Image):
    """
    エッジ検出を行う
    """
    # 8近傍ラプラシアンフィルタ
    # https://algorithm.joho.info/image-processing/laplacian-filter/
    _laplacian = np.array([[1,  1, 1],
                           [1, -8, 1],
                           [1,  1, 1]]).flatten()
    laplacian_kernel = ImageFilter.Kernel(size=(3,3), kernel=_laplacian, scale=1)

    img_edge = 255 - np.array(img.filter(ImageFilter.GaussianBlur(radius=1)).filter(laplacian_kernel))
    img_edge = np.where(img_edge <= 240, 0, 255).astype(np.uint8) # 2値化
    img_edge = Image.fromarray(img_edge)
    return img_edge 

def _gray2ternary(img: Image, bright: int = 20, grayzone: int = 40) -> Image:
    """
    gray画像から３値化する
    """
    _th = filters.threshold_otsu(np.array(img))
    th1 = _th - bright # 閾値1
    th2 = _th - bright+grayzone # 閾値2

    _tmp = np.array(img)
    _tmp = np.where(_tmp < th1, 0, _tmp)
    _tmp = np.where((th1 <= _tmp) & (_tmp < th2), 128, _tmp)
    result = np.where(th2 <= _tmp, 255, _tmp)
    return Image.fromarray(result)

def _spreading_screentone(img: Image, screen_tone: Image) -> Image:
    """
    img画像にトーンを貼る
    """
    # toneの大きさが足りないので、大きくする
    i = np.ceil(max(np.array(img.size)/np.array(screen_tone.size))).astype(int) # tone画像何枚分か計算
    tone_tile = np.tile(np.array(screen_tone), (i,i))[:img.size[1], :img.size[0]]
    img_array = np.array(img)
    img_tone = np.where(img_array == 128, tone_tile, img_array)
    return Image.fromarray(img_tone)

def make_cartoon_style(img: Image, screen_tone: Image, r: float=1, bright: int=20, grayzone: int=40) -> Image:
    img_gamma = _gamma(img, r) # 明るさを調節
    img_edge = _edge_detection(img_gamma) # エッジ検出(ラプラシアンフィルタ)
    img_ternary = _gray2ternary(img_gamma, bright, grayzone) # 3値化
    img_spreded_tone = _spreading_screentone(img_ternary, screen_tone) # トーンを貼る
    # 漫画風にする
    cartoon_style = np.where(np.array(img_edge) == 0, 0, np.array(img_spreded_tone))
    img_cartoon_style = Image.fromarray(cartoon_style)
    return img_cartoon_style

if __name__ == "__main__":
    
    args = len(sys.argv)
    
    if args < 2:
        print("第1引数には元画像のパスを入れておくれ")
        sys.exit(1)
    elif args < 3:
        print("第2引数にはトーン画像のパスを入れておくれ")
        sys.exit(1)
    
    # 使用する要素を初期化
    file_path = sys.argv[1]
    tone_path = sys.argv[2]
    r = sys.argv[3] if args > 3 else 1
    bright = sys.argv[4] if args > 4 else 20
    grayzone = sys.argv[5] if args > 5 else 40
    
    # 元画像とトーン画像の読み込み
    img = Image.open(file_path)
    tone = Image.open('data/screen.jpg').convert('L')
    
    # 漫画風にする処理
    img_cartoon_style = make_cartoon_style(img, tone, r, bright, grayzone)
    
    # 出力・保存
    img_cartoon_style.show()
    img_cartoon_style.save('cartoon_style.jpg', format='JPEG')
    