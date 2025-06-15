from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from basicsr.utils.download_util import load_file_from_url
import cv2
import os

def init_real_upscaler():
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, scale=4)
    weights_path = load_file_from_url(
        url='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth', 
        model_dir='weights', progress=True, file_name='RealESRGAN_x4plus.pth')
    return RealESRGANer(model_path=weights_path, model=model, scale=4, tile=400, half=False)

def upscale_image(input_path, label):
    try:
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        upscaler = init_real_upscaler()
        result, _ = upscaler.enhance(img, outscale=4)
        output_path = input_path.replace(".png", "_upscaled.png")
        cv2.imwrite(output_path, result)
        return output_path
    except Exception as e:
        print(f"Ошибка в upscale_image: {str(e)}")
        return None
