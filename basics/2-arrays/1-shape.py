import matplotlib.image as mpimg # import matpotlib's image module to plot image
import os # import os module for file location

dir_path = os.path.dirname(os.path.realpath(__file__)) # path of current directory
filename = os.path.join(dir_path, "MarshOrchid.jpg") # filename

image = mpimg.imread(filename) # read image file
print(image.shape) # print image shape
