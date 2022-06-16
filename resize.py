import os
from PIL import Image

f = r'C://Users/tpneo/OneDrive/Desktop/resize-me'
for file in os.listdir(f):
    f_img = f + "/" + file
    img = Image.open(f_img)
    img = img.resize((480, 480))
    img.save(f_img, optimize=True, quality=60)
