import cv2
import numpy as np

# Đọc ảnh từ file
src = cv2.imread('hinhanh.jfif', cv2.IMREAD_GRAYSCALE)

if src is None:
    print("Không thể mở ảnh!")
    exit()

# 1. Dò biên bằng toán tử Sobel
grad_x = cv2.Sobel(src, cv2.CV_16S, 1, 0, ksize=3)
grad_y = cv2.Sobel(src, cv2.CV_16S, 0, 1, ksize=3)

abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)

# Kết hợp theo công thức sqrt(Gx^2 + Gy^2)
grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

# Hiển thị kết quả của Sobel
cv2.imshow("Bien anh bang Sobel", grad)

# 2. Dò biên bằng Laplacian of Gaussian (LoG)
blurred = cv2.GaussianBlur(src, (5, 5), 0)
log_result = cv2.Laplacian(blurred, cv2.CV_16S, ksize=3)
log_result = cv2.convertScaleAbs(log_result)

# Hiển thị kết quả của Laplacian of Gaussian
cv2.imshow("Bien anh bang Laplacian of Gaussian", log_result)

cv2.waitKey(0)
cv2.destroyAllWindows()

