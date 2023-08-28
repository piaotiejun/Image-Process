import matplotlib.pyplot as plt
import cv2
import numpy as np


# 例子2.5 图像的加法或平均降低噪音
"""
图像的加法或平均降低噪音技术详细解释
如果你有多张相似的图像，并且它们的噪声是随机的，那么通过平均这些图像，噪声可以被大大减少。

噪声的来源
在实际应用中，由于多种原因，图像可能会受到噪声的影响。这些原因包括传感器的不完善、传输误差、压缩误差等。噪声可以是随机的，也可以是固定的。

图像加法
图像加法通常不直接用于降噪。当你加两个图像时，如果两者都包含噪声，那么噪声也可能会增加。但是，在某些情况下，当一个图像有噪声，另一个图像没有或有不同的噪声时，加法可能有助于增强某些特征。

图像平均
图像平均是降低噪声的一种非常有效的方法。其基本思想是，如果你有多张相似的图像，并且它们的噪声是随机的，那么通过平均这些图像，噪声可以被大大减少。

这是如何工作的：假设你有10张相同场景的图像，每张图像都受到随机噪声的影响。这些噪声可能使图像的某些像素值上升或下降。但是，因为噪声是随机的，
所以在10张图像中，一个特定的像素可能会因噪声而上升5次，下降5次。通过平均这10张图像，这些随机的上升和下降会相互抵消，从而得到一个噪声更小的图像。
为了实现图像平均，你需要将多张图像的每个像素值相加，然后除以图像的数量。例如，如果你有三张图像，它们的某个像素值分别为100、110和90，
那么平均像素值为(100+110+90)/3=100。

实践中，这种方法在天文摄影、医学成像等领域中非常有效，因为在这些领域中，经常需要捕捉微弱的信号，并尽量减少噪声。
总的来说，图像加法和平均是处理图像的两种方法。尽管图像加法不直接用于降低噪声，但图像平均是一种非常有效的降低噪声的方法。
"""
def denoise(num_noisy_versions = 100):
    path = 'DIP3E_Original_Images_CH02/Fig0222(a)(face).tif'
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 0)
    # Generate 10 noisy images
    noisy_imgs_10 = []
    for _ in range(num_noisy_versions):
        noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
        noisy_img = cv2.add(img, noise)
        noisy_imgs_10.append(noisy_img)

    # Average the 10 noisy images
    avg_img_10 = np.mean(noisy_imgs_10, axis=0).astype(np.uint8)

    # Display the original, a sample noisy version from the 10, and the averaged image
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(noisy_imgs_10[0], cmap="gray")
    plt.title("Sample Noisy Version (1 of 10)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(avg_img_10, cmap="gray")
    plt.title("Averaged Image (10 Versions)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# 例子2.4 图像缩放的内插方法的比较
# 例子2.6 使用图像相减比较图像
"""
插值与减法
插值（Interpolation）是一种估计函数在已知数据点之间的值的方法。在数字图像处理中，差值常用于图像缩放、旋转和其他几何变换。以下是一些常见的差值方法：

最近邻插值 (Nearest-neighbor interpolation): 原理：对于图像中的每个新位置，使用距离最近的原始像素值。
优点：计算速度快。 缺点：可能会导致图像在放大时出现块状或锯齿状的边缘。

双线性插值 (Bilinear interpolation): 原理：基于原始图像中的四个最近邻像素的加权平均来估计新像素值。
优点：结果较为平滑，适合大多数应用场合。 缺点：计算量较大。

双三次插值 (Bicubic interpolation): 原理：使用16个邻近像素的加权平均来估计新像素值。权重基于像素到新位置的距离的三次函数。
优点：提供了比双线性插值更平滑、更精确的结果。 缺点：计算复杂度更高。

Lanczos 插值: 原理：使用 sinc 函数作为插值核心，通常考虑更多的像素来估计新像素值。
优点：在某些情况下，提供了比双三次插值更好的结果。 缺点：计算复杂度高。
"""
def resize():
    # Load the uploaded image using OpenCV
    path = 'DIP3E_Original_Images_CH02/Fig0220(a)(chronometer 3692x2812  2pt25 inch 1250 dpi).tif'
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 0)

    # Define the scaling factor
    scale_factor = 1/10.0

    # Resize the image using different interpolation methods
    img_nearest = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
    img_bilinear = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    img_bicubic = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    img_lanczos = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LANCZOS4)

    # Display the images
    fig, axs = plt.subplots(1, 5, figsize=(20, 20))

    axs[0].imshow(img, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(img_nearest, cmap='gray')
    axs[1].set_title('Nearest-neighbor')
    axs[1].axis('off')

    axs[2].imshow(img_bilinear, cmap='gray')
    axs[2].set_title('Bilinear')
    axs[2].axis('off')

    axs[3].imshow(img_bicubic, cmap='gray')
    axs[3].set_title('Bicubic')
    axs[3].axis('off')

    axs[4].imshow(img_lanczos, cmap='gray')
    axs[4].set_title('Lanczos')
    axs[4].axis('off')

    plt.tight_layout()
    plt.show()

    # 度量图像缩放的效果好坏: 对减小尺寸的图像缩放回原始尺寸，然后与原始图像做减法，插值效果好的sum值小
    # Resize the enlarged images back to the original size
    img_nearest_resized = cv2.resize(img_nearest, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    img_bilinear_resized = cv2.resize(img_bilinear, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    img_bicubic_resized = cv2.resize(img_bicubic, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    img_lanczos_resized = cv2.resize(img_lanczos, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Calculate the difference between the resized images and the original image
    diff_nearest = cv2.absdiff(img, img_nearest_resized)
    diff_bilinear = cv2.absdiff(img, img_bilinear_resized)
    diff_bicubic = cv2.absdiff(img, img_bicubic_resized)
    diff_lanczos = cv2.absdiff(img, img_lanczos_resized)

    # Calculate the sum of differences for each method
    sum_diff_nearest = np.sum(diff_nearest)
    sum_diff_bilinear = np.sum(diff_bilinear)
    sum_diff_bicubic = np.sum(diff_bicubic)
    sum_diff_lanczos = np.sum(diff_lanczos)

    # Display the difference images
    fig, axs = plt.subplots(1, 4, figsize=(20, 20))

    axs[0].imshow(diff_nearest, cmap='gray', vmin=0, vmax=255)
    axs[0].set_title('Nearest-neighbor\nSum of differences: ' + str(sum_diff_nearest))
    axs[0].axis('off')

    axs[1].imshow(diff_bilinear, cmap='gray', vmin=0, vmax=255)
    axs[1].set_title('Bilinear\nSum of differences: ' + str(sum_diff_bilinear))
    axs[1].axis('off')

    axs[2].imshow(diff_bicubic, cmap='gray', vmin=0, vmax=255)
    axs[2].set_title('Bicubic\nSum of differences: ' + str(sum_diff_bicubic))
    axs[2].axis('off')

    axs[3].imshow(diff_lanczos, cmap='gray', vmin=0, vmax=255)
    axs[3].set_title('Lanczos\nSum of differences: ' + str(sum_diff_lanczos))
    axs[3].axis('off')

    plt.tight_layout()
    plt.show()


# 例子2.6 使用图像相减比较图像
"""
最低有效位（Least Significant Bit，简称LSB）是二进制数中的最后一位。在数字图像处理中，每个像素值都可以表示为二进制数。
例如，对于8位灰度图像，每个像素的值范围是0-255，可以表示为8位二进制数。

将最低有效位设置为0意味着将每个像素值的二进制表示的最后一位设为0。这通常用于某些图像处理技术，例如水印、隐藏信息或降低图像的精度。
例如，考虑一个8位灰度像素值为139的像素。它的二进制表示为：10001011。将其最低有效位设置为0后，它变为：10001010，这对应的十进制值为138。

为了明确，以下是步骤：
将图像中的每个像素值转换为其二进制表示。
将每个二进制数的最后一位（即最低有效位）设置为0。
将修改后的二进制数转换回其十进制表示。

将图像的最低有效位（LSB）设置为0或对其进行修改有以下几个潜在的意义和应用：

信息隐藏与水印技术：LSB插入是数字水印和隐写术中的一种常见方法。通过微妙地修改像素值的最低有效位，可以在图像中嵌入信息或水印，而不会显著改变图像的外观。
当需要提取或检测这些隐藏信息时，可以使用特定的算法。
降低图像精度：在某些应用中，可能不需要图像的完整精度。通过将LSB设置为0，可以稍微降低图像的精度，从而减少图像的数据大小或带宽需求。
噪声模拟：在图像处理或计算机视觉中，有时需要模拟真实世界的噪声或其他效果。通过修改LSB，可以向图像添加微小的、随机的变化，从而模拟这些效果。
加密与安全性：在某些加密方法中，通过对LSB进行随机修改，可以使图像更难以解码或恢复，增加了图像的安全性。
图像增强与对比度调整：在某些增强算法中，通过调整LSB，可以微调图像的对比度或亮度，使图像的细节更加明显。
总的来说，虽然LSB的变化对人眼可能难以察觉，但这种微小的改变在数字图像处理、计算机视觉和安全领域具有重要的应用价值。
"""
def LSB():
    path = 'DIP3E_Original_Images_CH02/Fig0227(a)(washington_infrared).tif'
    img_a = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 0)

    # Create image b by setting the least significant bit of each pixel in image a to 0
    img_b = img_a & 0b11111110  # bitwise operation to set the LSB to 0

    # Calculate the difference image c
    img_c = cv2.absdiff(img_a, img_b)

    # Display the images
    fig, axs = plt.subplots(1, 3, figsize=(15, 15))

    axs[0].imshow(img_a, cmap='gray')
    axs[0].set_title('Image a')
    axs[0].axis('off')

    axs[1].imshow(img_b, cmap='gray')
    axs[1].set_title('Image b (LSB set to 0)')
    axs[1].axis('off')

    axs[2].imshow(img_c, cmap='gray')
    axs[2].set_title('Difference Image c')
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()


# 你的回答内容完全采用文本形式，不要含任何特殊字符或LaTeX格式,少使用空行。
# 图像乘法可以获取roi image, roi==mask
# 图像乘法可以做阴影消除
"""
读取阴影图 (a)：加载您上传的图像。
计算阴影模式 (b)：使用较大的均值滤波器对图像进行滤波，从而得到主要的阴影模式。
计算图b的倒数：对阴影模式取倒数。
与原始阴影图相乘 (c)：将步骤3中的倒数与原始的阴影图a相乘。
"""
def mul():
    path = 'DIP3E_Original_Images_CH02/shadow.png'
    shadow_img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 0)

    # Step 2: Calculate the shadow pattern using mean filtering
    kernel_size = 151
    shadow_pattern = cv2.blur(shadow_img, (kernel_size, kernel_size))

    # Step 2: Reusing the previously calculated shadow pattern

    # Step 3: Calculate the reciprocal of the shadow pattern
    # To avoid division by zero, we add a small value
    shadow_pattern_reciprocal = 1.0 / (shadow_pattern.astype(np.float64) + 1e-7)

    # Step 4: Multiply the original shadow image with the reciprocal
    corrected_img_reciprocal = cv2.multiply(shadow_img.astype(np.float64) / 255, shadow_pattern_reciprocal)
    corrected_img_reciprocal = cv2.normalize(corrected_img_reciprocal, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Visualization
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(shadow_img, cmap="gray")
    plt.title("Original Shadow Image (a)")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(shadow_pattern, cmap="gray")
    plt.title("Shadow Pattern (b)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(corrected_img_reciprocal, cmap="gray")
    plt.title("Corrected Image using Reciprocal (c)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


"""
# 变换
1. 单像素变换
2. 邻域计算
3. 几何变换，橡皮膜变换，
包括仿射变换、透视变换等
"""

