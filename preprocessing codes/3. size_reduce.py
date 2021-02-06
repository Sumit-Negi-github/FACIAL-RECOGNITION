from PIL import Image,ImageOps
from skimage import io
import os

#Let the input image  is "3. cropped_image" and is stored in "sample_Image" folder.
im=Image.open("sample_Image/3. cropped_image.jpg")

im1=im.resize((50,50))
#save the size reduced image by name "4. size_reduced_image"  in folder "sample_Image"
im1.save("sample_Image/4. size_reduced_image.jpg")
