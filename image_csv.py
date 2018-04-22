from __future__ import with_statement
from PIL import Image
import numpy as np
from scipy import ndimage
import scipy
import matplotlib.pyplot as plt
 
im = Image.open('image.jpg') #relative path to file
def method_1():
    #load the pixel info
    pix = im.load()
    
    #get a tuple of the x and y dimensions of the image
    width, height = im.size
    
    #open a file to write the pixel data
    with open('output_file.csv', 'w+') as f:
      f.write('R,G,B\n')
    
      #read the details of each pixel and write them to the file
      for x in range(width):
        for y in range(height):
          r = pix[x,y][0]
          g = pix[x,x][1]
          b = pix[x,x][2]
          f.write('{0},{1},{2}\n'.format(r,g,b))

size = 28, 28

im_resized = im.resize(size, Image.ANTIALIAS)
im_resized.save("2.jpg", "JPEG")

my_image = "2.jpg"   # change this to the name of your image file 
## END CODE HERE ##
num_px=200
# We preprocess the image to fit your algorithm.
fname = my_image

image = np.array(ndimage.imread(fname, flatten=False))
#my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*4)).T
#my_predicted_image = predict(d["w"], d["b"], my_image)
print(image.shape)
plt.imshow(image)
plt.show()


