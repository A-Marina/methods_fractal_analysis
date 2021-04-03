# Импортируемые библиотеки
import numpy as np
import scipy
import PIL
from PIL import Image as img
import cv2
from google.colab.patches import cv2_imshow

# Печать изображения
i= img.open('im_3.jpg') # Название изображения
h, w = i.size
print (h,w)
i

# Вывод размера (высота и ширина)
img = cv2.imread("im_3.jpg") # Название изображения
h, w,c= img.shape
print ('height -',h, '\n widh-', w)


# Функции вычисления метода наименьших квадратов и фрактальной размерности
def box_count(Z, size):
    S = np.add.reduceat(
        np.add.reduceat(Z, np.arange(0, Z.shape[0], size), axis=0),
        np.arange(0, Z.shape[1], size), axis=1)
    count = 0
    for x in range(S.shape[0]):
        for y in range(S.shape[1]):
            if S[x][y] > 0 & S[x][y] < size * size:
                count += 1;
    return count


def fractal_dimension(Z, threshold=0.6):
    Z = (Z < threshold)
    a = min(Z.shape)
    n1 = 2 ** np.floor(np.log(a) / np.log(2))
    n2 = int(np.log(n1) / np.log(2))
    sizes = 2 ** np.arange(n2, 1, -1)
    counts = []
    for size in sizes:
        counts.append(box_count(Z, size))
    # МНК
    leastSquares = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -leastSquares[0]


# Вызов функции вычисления размерности
original = cv2.imread('im_3.jpg')

grayIm = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
threshold=cv2.threshold(grayIm, 127, 255, cv2.THRESH_BINARY)[0] / 255
print("фрактальная размерность: ", fractal_dimension(grayIm, threshold))
