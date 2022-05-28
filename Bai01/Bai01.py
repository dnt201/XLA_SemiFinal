from PIL import Image
IMG_PATH = ('pic01.jpg')

img = Image.open(IMG_PATH)
pixels = img.load()
constant = int(input('Input constant: '))

new_img = Image.new(img.mode, img.size)
pixels_new = new_img.load()
for i in range(new_img.size[0]):
    for j in range(new_img.size[1]):
        r, b, g = pixels[i,j]
        pixels_new[i,j] = (r * constant, b * constant, g * constant, 0)
new_img.show()