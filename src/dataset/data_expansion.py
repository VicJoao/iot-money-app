import imgaug as ia
import imgaug.augmenters as iaa
import cv2
from glob import glob
import numpy as np

# Define a lambda function for applying augmenters sometimes
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define the augmentation sequence using imgaug.Sequential
seq = iaa.Sequential(
    [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.2),
        sometimes(iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-45, 45),
            shear=(-16, 16),
            order=[0, 1],
            cval=(0, 255),
            mode=ia.ALL
        )),
        iaa.SomeOf((0, 5),
                   [
                       sometimes(iaa.BlendAlphaSimplexNoise(iaa.OneOf([
                           iaa.EdgeDetect(alpha=(0.5, 1.0)),
                           iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                       ]))),
                       iaa.OneOf([
                           iaa.GaussianBlur((0, 3.0)),
                           iaa.AverageBlur(k=(2, 7)),
                           iaa.MedianBlur(k=(3, 11)),
                       ]),
                       iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                       iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
                       sometimes(iaa.BlendAlphaFrequencyNoise(
                           exponent=(-4, 0),
                           foreground=iaa.Multiply((0.5, 1.5), per_channel=True),
                           background=iaa.LinearContrast((0.5, 2.0))
                       )),
                       iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                       iaa.OneOf([
                           iaa.Dropout((0.01, 0.1), per_channel=0.5),
                           iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                       ]),
                       iaa.Invert(0.05, per_channel=True),
                       iaa.Add((-10, 10), per_channel=0.5),
                       iaa.AddToHueAndSaturation((-20, 20)),
                       iaa.OneOf([
                           iaa.Multiply((0.5, 1.5), per_channel=0.5),
                           sometimes(iaa.BlendAlphaFrequencyNoise(
                               exponent=(-4, 0),
                               foreground=iaa.Multiply((0.5, 1.5), per_channel=True),
                               background=iaa.LinearContrast((0.5, 2.0))
                           )),
                       ]),
                       iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
                       iaa.Grayscale(alpha=(0.0, 1.0)),
                       sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                       sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                       sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                   ],
                   random_order=True
                   )
    ],
    random_order=True
)

import os

output_path = os.path.join(os.path.dirname(__file__), 'model', 'DataSetExp')
folder_path = os.path.join(os.path.dirname(__file__), 'model', 'DataSet')
n_gen = 2
print("ok")

for filename in os.listdir(folder_path):
    if filename.endswith('.jpg'):
        img_path = os.path.join(folder_path, filename)
        print("ok")

        try:
            image = cv2.imread(img_path)
            filename_without_extension = os.path.splitext(filename)[0]

            for i in range(n_gen):
                images_aug = seq(images=[image])
                output_filename = f'{filename_without_extension}_{i}.jpg'
                output_filepath = os.path.join(output_path, output_filename)
                cv2.imwrite(output_filepath, images_aug[0])
                print('.', end='', flush=True)  # Print a dot for each successful image processed

        except Exception as e:
            print(f"Failed to process {img_path}: {str(e)}")

