from keras import backend as K
from keras.preprocessing import image
from keras.applications.resnet import preprocess_input, decode_predictions
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims
import inception_v4
import numpy as np
import cv2
import os


# If you want to use a GPU set its index here
os.environ['CUDA_VISIBLE_DEVICES'] = ''


# This function comes from Google's ImageNet Preprocessing Script
def central_crop(image, central_fraction):
        """Crop the central region of the image.
        Remove the outer parts of an image but retain the central region of the image
        along each dimension. If we specify central_fraction = 0.5, this function
        returns the region marked with "X" in the below diagram.
           --------
          |        |
          |  XXXX  |
          |  XXXX  |
          |        |   where "X" is the central 50% of the image.
           --------
        Args:
        image: 3-D array of shape [height, width, depth]
        central_fraction: float (0, 1], fraction of size to crop
        Raises:
        ValueError: if central_crop_fraction is not within (0, 1].
        Returns:
        3-D array
        """
        if central_fraction <= 0.0 or central_fraction > 1.0:
                raise ValueError('central_fraction must be within (0, 1]')
        if central_fraction == 1.0:
                return image

        img_shape = image.shape
        depth = img_shape[2]
        fraction_offset = int(1 / ((1 - central_fraction) / 2.0))
        bbox_h_start = int(np.divide(img_shape[0], fraction_offset))
        bbox_w_start = int(np.divide(img_shape[1], fraction_offset))

        bbox_h_size = int(img_shape[0] - bbox_h_start * 2)
        bbox_w_size = int(img_shape[1] - bbox_w_start * 2)

        image = image[bbox_h_start:bbox_h_start+bbox_h_size, bbox_w_start:bbox_w_start+bbox_w_size]
        return image


def get_processed_image(img_path):
        # Load image and convert from BGR to RGB
        im = np.asarray(cv2.imread(img_path))[:,:,::-1]
        im = central_crop(im, 0.875)
        im = cv2.resize(im, (299, 299))
        im = inception_v4.preprocess_input(im)
        if K.image_data_format() == "channels_first":
                im = np.transpose(im, (2,0,1))
                im = im.reshape(-1,3,299,299)
        else:
                im = im.reshape(-1,299,299,3)
        return im


if __name__ == "__main__":
        # Create model and load pre-trained weights
        model = inception_v4.create_model(weights='imagenet', include_top=True)

        ixs = [1,4,7,381,479]
        # blockConv = ["Conv_1","Conv_2", "Conv_3", "Conv_4", "Conv_5"]
        outputs = [model.layers[i].output for i in ixs]
        model = Model(inputs=model.inputs, outputs=outputs)

        # Open Class labels dictionary. (human readable label given ID)
        # classes = eval(open('validation_utils/class_names.txt', 'r').read())

        # Load test image!
        img_path = 'elephant.jpg'
        img = get_processed_image(img_path)

        preds = model.predict(img)

        square = 5
        pyplot.rcParams['figure.figsize'] = (20,20)
        count = 0
        for fmap in preds:
          # print("Block - {}".format(blockConv[count]))
          count += 1
          # plot all 64 maps in an 8x8 squares
          ix = 1
          for _ in range(square):
            for _ in range(square):

              # specify subplot and turn of axis
              ax = pyplot.subplot(square, square, ix)
              ax.set_xticks([])
              ax.set_yticks([])
              # plot filter channel in grayscale
              pyplot.imshow(fmap[0, :, :, ix-1], cmap='gray')
              ix += 1
          # show the figure
          pyplot.show()

        # # Run prediction on test image
        # # preds = model.predict(img)
        # print("Class is: " + classes[np.argmax(preds)-1])
        # print("Certainty is: " + str(preds[0][np.argmax(preds)]))