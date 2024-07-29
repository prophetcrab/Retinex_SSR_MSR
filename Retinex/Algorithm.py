import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def single_scale_retinex(img, sigma):
    img = np.float32(img) + 1.0
    log_img = np.log(img)
    gauss = cv2.GaussianBlur(img, (0, 0), sigma)
    log_gauss = np.log(gauss)
    retinex = log_img - log_gauss
    return cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def apply_ssr_to_image(image_path, sigma):
    img = cv2.imread(image_path)
    channels = cv2.split(img)
    result_channels = [single_scale_retinex(channel, sigma) for channel in channels]
    result_img = cv2.merge(result_channels)
    result_img = white_balance(result_img)
    return result_img

def multi_scale_retinex(img, scales):
    img = np.float32(img) + 1.0
    log_img = np.log(img)
    retinex = np.zeros_like(log_img)
    for sigma in scales:
        gauss = cv2.GaussianBlur(img, (0, 0), sigma)
        log_gauss = np.log(gauss)
        retinex += log_img - log_gauss
    retinex = retinex / len(scales)
    return cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def apply_msr_to_image(image_path, scales):
    img = cv2.imread(image_path)
    channels = cv2.split(img)
    result_channels = [multi_scale_retinex(channel, scales) for channel in channels]
    result_img = cv2.merge(result_channels)
    result_img = white_balance(result_img)
    return result_img

def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(result)
    l = cv2.equalizeHist(l)
    result = cv2.merge((l, a, b))
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

def save_image(image, save_path):
    cv2.imwrite(save_path, image)
    print("Image saved successfully")

if __name__ == '__main__':
    image_path = r'D:\PythonProject2\Retinex\data\image4.jpg'
    output_path = r'D:\PythonProject2\Retinex\result'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # SSR
    sigma = 300
    ssr_img = apply_ssr_to_image(image_path, sigma)
    ssr_img_rgb = cv2.cvtColor(ssr_img, cv2.COLOR_BGR2RGB)
    #cv2.imshow('SSR Image', ssr_img_rgb)
    save_image(ssr_img, os.path.join(output_path, 'SSR_Image.jpg'))

    # MSR
    scales = [200, 300, 400, 500, 600, 700]
    msr_img = apply_msr_to_image(image_path, scales)
    msr_img_rgb = cv2.cvtColor(msr_img, cv2.COLOR_BGR2RGB)
    #cv2.imshow('MSR Image', msr_img_rgb)
    save_image(msr_img, os.path.join(output_path, 'MSR_Image.jpg'))

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 显示图片
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(ssr_img_rgb)
    plt.title('SSR Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(msr_img_rgb)
    plt.title('MSR Image')
    plt.axis('off')

    plt.show()

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
