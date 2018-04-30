from PIL import Image
im = Image.open('nine_small.png')
im.load() #required unless you want to close the image
pixels = list(im.getdata())
width, height = im.size
# pixels [(r,g,b,a), (r,g,b,a), (r,g,b,a)]
# brightness is just the weighted average of all pixels, can also just 
# normall average them
print(pixels)
print(width, height)
