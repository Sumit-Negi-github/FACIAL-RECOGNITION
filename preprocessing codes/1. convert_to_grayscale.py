from PIL import Image,ImageOps
from skimage import io
import os

#Let the input image  is "1. photo" and is stored in "sample_Image" folder.
im=Image.open("sample_Image/1. photo.png")
im1=ImageOps.grayscale(im)

#save the grayscaled image by name "2. grayscaled_image"  in folder "sample_Image"
im1.save("sample_Image/2. grayscaled_image.jpg")
