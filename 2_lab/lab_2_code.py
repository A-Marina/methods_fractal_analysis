import numpy as np
from PIL import Image
import itertools
import matplotlib.pyplot as plt
import cv2

img = Image.open('image_2.jpg') # Название изображения
img = img.convert('RGB')
colors = img.getcolors(maxcolors=1000000)
colors=sorted(colors, reverse=True)

a=colors[:2]
f,s =a
for amount, color in a:
  print('цвет -', f'{color} в RGB. Повторяется {amount} раз (пикселей)')
print('')

# Вывод палитры
f1,f2=f
s1,s2=s
print('Палитра:')
palette=[f2,s2,t2]
palette = np.array(palette)[np.newaxis, :, :]
plt.imshow(palette);
plt.axis('off');
plt.show();

# Печать изображения
i= Image.open('image_2.jpg') # Название изображения 
h, w = i.size
print (h,w)
i

