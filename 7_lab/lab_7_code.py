# Импортируемые библиотеки
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Ввод названия изображения
image_name = 'img.jpg'

# Вывод исходного изображения
i= Image.open(image_name) 
h, w = i.size
print (h,w)
i

# Вспомогательные функции вычисления u, b, v_ol и A
def u_d(img, max_delta):
    width, height = img.shape
    u = np.zeros((max_delta + 1, width + 1, height + 1))
    for i in range(0, width):
        for j in range(0, height):
            u[0][i][j] = img[i][j]
            for d in range(1, max_delta + 1):
                u[d][i][j] = max(u[d-1][i][j] + 1, max(u[d-1][i+1][j+1], u[d-1][i-1][j+1], u[d-1][i+1][j-1], u[d-1][i-1][j-1]))
    return u

def b_d(img, max_delta):
    width, height = img.shape
    b = np.zeros((max_delta + 1, width + 1, height + 1))
    for i in range(0, width):
        for j in range(0, height):
            b[0][i][j] = img[i][j]
            for d in range(1, max_delta + 1):
                b[d][i][j] = min(b[d-1][i][j] - 1, min(b[d-1][i+1][j+1], b[d-1][i-1][j+1], b[d-1][i+1][j-1], b[d-1][i-1][j-1]))
    return b

def v_ol(img, max_delta):
    width, height = img.shape
    u = u_d(img, max_delta)
    b = b_d(img, max_delta)
    vol = np.zeros(max_delta + 1)
    for d in range(1, max_delta + 1):
        sum = 0
        for i in range(0, width):
            for j in range(0, height):
                sum += u[d][i][j] - b[d][i][j]
        vol[d] = sum
    return vol

def get_A(img, max_delta):
    # u1 = u_s(img)
    # u2 = u_s(u1)
    # u3 = u_s(u2)
    # b1 = b_s(img)
    # b2 = b_s(b1)
    # b3 = b_s(b2)
    # vol1 = v_ol(u1, b1)
    # vol2 = v_ol(u2, b2)
    # vol3 = v_ol(u3, b3)
    # A_2 = (vol2 - vol1) / 2
    # A_3 = (vol3 - vol2) / 2
    # return (np.log(A_2) - np.log(A_3))
    A = np.zeros(max_delta + 1)
    vol = v_ol(img, max_delta)
    for d in range(1, max_delta + 1):
        A[d] = (vol[d] - vol[d-1]) / 2
    return A

  
# Вычисление размерности
def get_D(img, max_delta, cell_size):
    width, height = img.shape
    cell_width = int(width / cell_size)
    cell_heigth = int(height / cell_size)
    A = np.zeros((cell_width, cell_heigth, max_delta + 1))
    for i in range(0, cell_width):
        for j in range(0, cell_heigth):
            cell_img = img[j*cell_size:cell_size * (j+1), i*cell_size:cell_size*(i+1)]
            a = get_A(cell_img, max_delta)
            for d in range(1, max_delta + 1):
                A[i][j][d] = a[d]
    sum_A = []
    for d in range(1, max_delta + 1):
        sum = 0
        for i in range(0, cell_width):
            for j in range(0, cell_heigth):
                sum += A[i][j][d]
        sum_A.append(sum)
    ss = np.polyfit(np.log(sum_A), np.log(np.arange(1, max_delta+1)), 1)[0] * (-1)
    return 2 - ss
	
	
	
# Построение графика

# plt.xlabel('размер ячейки')
# plt.ylabel('размерность')
# l_s = []
# l, = plt.plot(sizes, D)
# l_s.append(l)
# plt.legend(l_s, image_name)
# plt.savefig('result_'+image_name)
fig, ax = plt.subplots()
res = []
cell_range = range(10, 100, 5)
for cell in cell_range:
    img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    res.append(get_D(img, 2, cell))
ax.plot(cell_range, res)
ax.set(xlabel='размер ячейки', ylabel='размерность')
ax.grid()
fig.savefig("result_"+image_name)
plt.show()