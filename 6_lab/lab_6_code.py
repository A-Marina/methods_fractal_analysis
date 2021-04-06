# Импортируемые библиотеки
import numpy as np
import scipy
import cv2
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
from PIL import Image

# Ввод названия изображения
image_name = 'image.jpg'

# Вывод исходного изображения
i= Image.open(image_name) 
h, w = i.size
print (h,w)
i

# Вспомогательные функции вычисления u_s, b_s, v_ol и A
def u_s(img):
  k,l=img.shape
  u=np.ndarray(img.shape)
  for i in range(k-1):
    for j in range(l-1):
      u[i][j]=max(img[i][j]+1,max(img[i-1][j],img[i][j-1],img[i+1][j],img[i][j+1]))
  return u

def b_s(img):
  k,l=img.shape
  b=np.ndarray(img.shape)
  for i in range(k-1):
    for j in range(l-1):
      b[i][j]=min(img[i][j]-1,min(img[i-1][j],img[i][j-1],img[i+1][j],img[i][j+1]))
  return b

def v_ol(u,b):
  k,l=u.shape
  vol=0.0
  for i in range(k):
    for j in range(l):
      vol=vol+u[i][j]-b[i][j]
  return vol

def get_A(vol2,vol1,beta):
  A = (vol2 - vol1) / 2
  lAlb = np.log(A) / np.log(beta)
  return lAlb

A_b=[]

  
# Вычисление вектора и построение графика
def plot_(betas, Div):
  plt.figure(figsize=(10, 10))
  plt.xlabel('b')
  plt.ylabel('log(A) / log(b)')
  line, = plt.plot(betas, Div)
  l_s.append(line)
  labels.append(image_name)
  plt.legend(l_s, labels)
  plt.savefig('result_'+image_name)

def get_div(betas):
  u_s_arr.append(u_s(u_s_arr[-1]))
  b_s_arr.append(b_s(b_s_arr[-1]))
  for beta in betas:
      u_s_arr.append(u_s(u_s_arr[-1]))
      b_s_arr.append(b_s(b_s_arr[-1]))
      vol1 = v_ol(u_s_arr[-2], b_s_arr[-2])
      vol2 = v_ol(u_s_arr[-1], b_s_arr[-1])
      lAlb = get_A(vol2,vol1,beta)
      A_b.append(lAlb)
      yield lAlb

l_s = []
labels = []
image = cv2.imread(image_name)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
u_s_arr = [image]
b_s_arr = [image]
betas = [i for i in range(1, 11)]
Div = [i for i in get_div(betas)]
plot_(betas,Div)

A_b
