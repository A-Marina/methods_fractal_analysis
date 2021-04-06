# Импортируемые библиотеки
import numpy as np
import scipy
import cv2
import matplotlib.pyplot as plt
from pandas import Series
from google.colab.patches import cv2_imshow
import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color
import skimage
from skimage import io
from PIL import Image


# Ввод названия изображения
image_name = 'image_2.jpg'


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
  v_ol=0.0
  for i in range(k):
    for j in range(l):
      v_ol=v_ol+u[i][j]-b[i][j]
  return v_ol

def get_A(gImage):
  u_1=u_s(gImage)
  u_2=u_s(u_1)
  b_1=b_s(gImage)
  b_2=b_s(b_1)
  v_ol_1=v_ol(u_1,b_1)
  v_ol_2=v_ol(u_2,b_2)
  A=(v_ol_2-v_ol_1)/(2)
  return A
  
  
# Сегментация и вывод
def segm(img):
    segmented_img = np.full(img.shape, 255)
    A_s = []
    for i in range(0, img.shape[0], 5):
        for j in range(0, img.shape[1], 5):
            A = get_A(img[i:i + 5, j: j + 5])
            A_s.append(A)
            if A >= 1000:
                segmented_img[i:i + 5,
                j:j + 5].fill(0)
    cv2.imwrite('result_'+image_name, segmented_img)
    
def image_show(image, nrows=1, ncols=1):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    return fig, ax


originalImage = cv2.imread(image_name)
grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
segm(grayImage)
text = io.imread('result_'+image_name)
image_show(text)
