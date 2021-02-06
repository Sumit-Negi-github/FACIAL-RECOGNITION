from PIL import Image,ImageOps
from skimage import io
import os

#Let the input image  is "2. grayscaled_image.jpg" and is stored in "sample_Image" folder.
im=Image.open("sample_Image/2. grayscaled_image.jpg")

left=50
right=205
top=25
bottom=200
im1=im.crop((left,top,right,bottom))

#save the cropped image by name "3. cropped_image"  in folder "sample_Image"
im1.save("sample_Image/3. cropped_image.jpg")



