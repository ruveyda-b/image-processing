import cv2
import numpy as np

def classify_shape(contour, corners):
    if is_circle(contour):
        return "Cember"
    # koseleri sayma islemi
    corner_count = 0
    for corner in corners:
        x, y = corner.ravel() # köşe noktasını düzleştirir ve x, y koordinatlarını alır
        point = (int(x), int(y)) # noktayı (x, y) tuple formatına dönüştürür
        if cv2.pointPolygonTest(contour, point, False) >= 0: # bu fonksiyon, noktanın konturun nerede olduğunu belirler
            corner_count += 1                                           # sıfıra eşit veya büyükse nokta konturun içindedir ve kose sayısı arttırılır
    if corner_count == 3:
        return "Ucgen"
    elif corner_count == 4:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        if 0.95 <= aspect_ratio <= 1.05:
            return "Kare"
        else:
            return "Dikdortgen"
    elif corner_count > 4:
        return "Cokgen"
    else:
        return "Bilinmiyor"

def is_circle(contour):
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)
    circle_area = np.pi * (radius ** 2)
    contour_area = cv2.contourArea(contour)
    contour_perimeter = cv2.arcLength(contour, True)
    circularity = (4 * np.pi * contour_area) / (contour_perimeter ** 2)
    area_ratio = contour_area / circle_area
    return 0.8 < circularity < 1.2 and 0.7 < area_ratio < 1.3

# Resmi yükleme
image = cv2.imread("shapes.jpeg")

# Resmi küçültme
image = cv2.pyrDown(image)

# Resmi bulanıklaştırma
image = cv2.GaussianBlur(image, (5, 5), 0)
image = cv2.bilateralFilter(image, 9, 75, 75)

# Resmi griye çevirme
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Adaptive thresholding kullanarak binarization
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Morfolojik işlemlerle gürültüyü temizleme
kernel = np.ones((3, 3), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Konturları bulma (yalnızca dış konturlar)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Gürültü konturlarını filtreleme ve konturları çizme
min_area = 100  # Minimum kontur alanı
filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]

# Shi-Tomasi köşe algılama
corners_general = cv2.goodFeaturesToTrack(gray, maxCorners=50, qualityLevel=0.05, minDistance=20)
corners_general = np.int32(corners_general)

# Her konturun köşe sayılarını belirleme ve sınıflandırma
for contour in filtered_contours:
    shape = classify_shape(contour, corners_general)
    x, y, w, h = cv2.boundingRect(contour)
    cv2.putText(image, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.drawContours(image, [contour], -1, (0, 255, 255), 2)

    if shape == "Cember":
        # Çemberler için daha fazla köşe algılama
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        more_corners = cv2.goodFeaturesToTrack(mask, maxCorners=100, qualityLevel=0.04, minDistance=5)
        if more_corners is not None:
            more_corners = np.int32(more_corners)
            for corner in more_corners:
                x, y = corner.ravel()
                cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
    else:
        # Diğer şekillerin köşe noktalarını işaretleme
        for corner in corners_general:
            x, y = corner.ravel()
            if cv2.pointPolygonTest(contour, (int(x), int(y)), False) >= 0:
                cv2.circle(image, (x, y), 5, (255, 0, 0), -1)

cv2.imshow("Şekiller", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
