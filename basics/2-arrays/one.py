import matplotlib.image as mpimg # import matpotlib's image module to plot image
import os # import os module for file location

dir_path = os.path.dirname(os.path.realpath(__file__))
filename = os.path.join(dir_path, "MarshOrchid.jpg")

image = mpimg.imread(filename)
print(image.shape)
