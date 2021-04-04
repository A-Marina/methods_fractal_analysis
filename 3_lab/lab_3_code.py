# Импортируемы библиотеки
from PIL import Image, ImageDraw

# Вывод исходного изображения
name_of_image = 'image_2.jpg'
i= Image.open(name_of_image) 
h, w = i.size
print (h,w)
i


# Загрузка
name_of_image = 'image_2.jpg' 
orig = Image.open(name_of_image)
orig = orig.convert('RGB')
draw = ImageDraw.Draw(orig)
w, h = orig.size	
new_image = orig.load()


# Преобразование
for i in range(w):
		for j in range(h):
			red = new_image[i, j][0]
			green = new_image[i, j][1]
			blue = new_image[i, j][2]
			D = (red + green + blue) // 3
			red = D
			green = D
			blue = D+60
			if (red > 255):
				red = 255
			if (green > 255):
				green = 255
			if (blue > 255):
				blue = 255
			draw.point((i, j), (red, green, blue))
img_2 = Image.open(name_of_image).convert('LA')
			

# Вывод и сохранение результатов
new_name_blue = name_of_image.split('.')[0] + '_result_blue.' + name_of_image.split('.')[1]
new_name_gray = name_of_image.split('.')[0] + '_result_gray.png'
orig.save(new_name_blue, "JPEG")
img_2.save(new_name_gray)
orig
