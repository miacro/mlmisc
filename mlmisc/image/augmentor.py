import argparse
import os
import glob
import cv2
from imgaug import augmenters as iaa
import imgaug as ia


def load_images(imagedir):
    imagedir = glob.escape(imagedir)
    for extension in ("*.png", "*.jpeg", "*.jpg"):
        for filename in glob.glob(os.path.join(imagedir, extension)):
            image = cv2.imread(filename, cv2.IMREAD_COLOR)
            if image is None:
                continue
            yield image


def batch_images(images, batch_size=1):
    batch = []
    for image in images:
        batch.append(image)
        if len(batch) == batch_size:
            yield batch
            batch.clear()
    if batch:
        yield batch


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="image augmentor")
    parser.add_argument(
        "--image-dir", dest="imagedir", default="", required=True)
    parser.add_argument("--output-dir", dest="outputdir", default="output")
    args = parser.parse_args()
    seq = iaa.Sequential([
        iaa.Crop(px=(0, 16)),
        iaa.Fliplr(0.5),
        iaa.GaussianBlur(sigma=(0, 3.0))
    ])

    def sometimes(aug):
        return iaa.Sometimes(0.5, aug)

    seq = iaa.Sequential(
        [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.2),
            sometimes(iaa.Crop(percent=(0, 0.1))),
            sometimes(
                iaa.Affine(
                    scale={"x": (0.8, 1.2),
                           "y": (0.8, 1.2)},
                    translate_percent={"x": (-0.2, 0.2),
                                       "y": (-0.2, 0.2)},
                    rotate=(-45, 45),
                    shear=(-16, 16),
                    order=[0, 1],
                    cval=(0, 255),
                    mode=ia.ALL)),
            iaa.SomeOf(
                (0, 5), [
                    sometimes(
                        iaa.Superpixels(
                            p_replace=(0, 1.0), n_segments=(20, 200))),
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)),
                        iaa.AverageBlur(k=(2, 7)),
                        iaa.MedianBlur(k=(3, 11)),
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
                    sometimes(
                        iaa.OneOf([
                            iaa.EdgeDetect(alpha=(0, 0.7)),
                            iaa.DirectedEdgeDetect(
                                alpha=(0, 0.7), direction=(0.0, 1.0)),
                        ])),
                    iaa.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel=0.5),
                        iaa.CoarseDropout(
                            (0.03, 0.15),
                            size_percent=(0.02, 0.05),
                            per_channel=0.2),
                    ]),
                    iaa.Invert(0.05, per_channel=True),
                    iaa.Add((-10, 10), per_channel=0.5),
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                    iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                    sometimes(
                        iaa.ElasticTransformation(
                            alpha=(0.5, 3.5), sigma=0.25)),
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
                ],
                random_order=True)
        ],
        random_order=True)

    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)
    for batch_index, batch in enumerate(
            batch_images(load_images(args.imagedir), batch_size=20)):
        for index, image in enumerate(seq.augment_images(batch)):
            imagefile = os.path.join(args.outputdir,
                                     "{:0>6d}{:0>2d}-image.png".format(
                                         batch_index, index))
            print("saving {}".format(imagefile))
            cv2.imwrite(imagefile, image)


if __name__ == "__main__":
    main()
