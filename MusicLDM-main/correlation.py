import numpy as np
import cv2
from scipy import signal

def calculate_image_correlation(image1, image2):
    """
    2つの画像の相関値を計算する関数

    Args:
        image1 (numpy.ndarray): 最初の画像
        image2 (numpy.ndarray): 2番目の画像

    Returns:
        float: 画像間の相関値
    """
    # グレースケールに変換
    if len(image1.shape) > 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    if len(image2.shape) > 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # 画像サイズを揃える
    min_height = min(image1.shape[0], image2.shape[0])
    min_width = min(image1.shape[1], image2.shape[1])

    image1 = image1[:min_height, :min_width]
    image2 = image2[:min_height, :min_width]

    # 相関係数を計算
    correlation = signal.correlate2d(image1, image2, mode='same')

    # 正規化相関係数
    normalized_correlation = np.corrcoef(image1.flatten(), image2.flatten())[0, 1]

    return normalized_correlation

# 使用例
def main():
    # 画像を読み込む
    img1 = cv2.imread('Figure_1motoonngenn.png')
    img2 = cv2.imread('timestep600_re.png')
    print(img1.shape)
    # 相関値を計算
    correlation = calculate_image_correlation(img1, img2)

    # 結果を表示
    print(f"画像間の相関値: {correlation}")

if __name__ == "__main__":
    main()
