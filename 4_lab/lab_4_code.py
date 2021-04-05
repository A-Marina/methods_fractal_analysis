# Импортируемые библиотеки
import numpy as np
import scipy
import cv2
from google.colab.patches import cv2_imshow
from PIL import Image

# Вывод исходного изображения
i= Image.open('im_2.jpg') # Название изображения 
h, w = i.size
print (h,w)
i

# Вспомогательные функции вычисления u_s, b_s и v_ol
def v_ol(u,b):
  k,l=u.shape
  vol=0.0
  for i in range(k):
    for j in range(l):
      vol=vol+u[i][j]-b[i][j]
  return vol
  
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
  

# Подстчет u_s, b_s и v_ol и А
originalImage = cv2.imread('im_2.png')
grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
u1=u_s(grayImage)
u2=u_s(u1)
b1=b_s(grayImage)
b2=b_s(b1)
vol1=v_ol(u1,b1)
vol2=v_ol(u2,b2)
As1=vol1/2
As2=vol2/4


# Вычисление размерности и вывод результата
d=2-(np.log(As1)-np.log(As2))/(np.log(1)-np.log(2))
print(d)
