from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
def load_and_preprocess_image(image_path, target_size=(128, 128)):
    """
    画像を読み込み、指定されたサイズにリサイズし、1次元のベクトルに変換します。
    
    Args:
        image_path (str): 画像ファイルのパス
        target_size (tuple): 画像のリサイズ先のサイズ (width, height)
        
    Returns:
        np.array: 1次元のベクトル化された画像データ
    """
    image = Image.open(image_path).convert('RGB')  # 画像をRGB形式で開く
    image = image.resize(target_size)  # 画像をリサイズ
    image_array = np.array(image)  # NumPy配列に変換
    flattened_array = image_array.flatten()  # 1次元ベクトルに変換
    normalized_array = flattened_array / 255.0  # [0, 255] を [0, 1] に正規化
    return normalized_array


def calculate_ssim(image_path1, image_path2):
    #画像を読み込み（グレースケール）
    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
    print(img1.shape)
    print(img2.shape)
    #入力チェック
    if img1 is None or img2 is None:
        print("画像の読み込みに失敗しました。ファイルパスを確認してください。")
        return
    
    #画像サイズのチェック
    if img1.shape != img2.shape:
        print("画像サイズが一致していません。画像をリサイズしてください。")
        return
    
    #SSIMの計算
    ssim_value, _ = ssim(img1, img2, full=True)
    print(f"SSIM: {ssim_value:.4f}")

#使用例
#画像ファイルのパスを指定
image1_path = "Figure_1motoonngenn.png"
image2_path = "Figure_1tanbunn.png"

# 画像をベクトル化
image_vector1 = load_and_preprocess_image(image1_path)
image_vector2 = load_and_preprocess_image(image2_path)

# 2つの画像ベクトルを(1, N)の形に変換してコサイン類似度を計算
cosine_sim = cosine_similarity([image_vector1], [image_vector2])

print(f"2つの画像のコサイン類似度: {cosine_sim[0][0]:.4f}")
calculate_ssim(image1_path, image2_path)


