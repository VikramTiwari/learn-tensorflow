import matplotlib.image as mpimg
import matplotlib.pyplot as plt # used to plot data
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
filename = os.path.join(dir_path, 'MarshaOrchid.jpg')

image = mpimg.imread(filename)
plt.imshow(image) # plot image
plt.show() # show the resulting plot which in this case will be an image
