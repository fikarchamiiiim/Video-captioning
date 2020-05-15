from PIL import Image
import numpy as np

w, h = 224, 224
# data = np.zeros((h, w, 3), dtype=np.uint8)
data = np.load("../data/feats/resnet152/video0.npy")
print(data.shape)
img = Image.fromarray(data)
# img.save('my.png')
img.show()