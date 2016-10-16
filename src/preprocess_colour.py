import cv2
import errno
import numpy as np
import math
import sys
import os
import getopt
from joblib import Parallel, delayed
import multiprocessing


def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()

def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix

def simplest_cb(img, percent):
    assert img.shape[2] == 3
    assert percent > 0 and percent < 100

    half_percent = percent / 200.0

    channels = cv2.split(img)

    out_channels = []
    for channel in channels:
        assert len(channel.shape) == 2
        # find the low and high precentile values (based on the input percentile)
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)

        assert len(flat.shape) == 1

        flat = np.sort(flat)

        n_cols = flat.shape[0]

        low_val  = flat[int(math.floor(n_cols * half_percent))]
        high_val = flat[int(math.ceil( n_cols * (1.0 - half_percent)))]

        # print "Lowval: ", low_val
        # print "Highval: ", high_val

        # saturate below the low percentile and above the high percentile
        thresholded = apply_threshold(channel, low_val, high_val)
        # scale the channel
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    return cv2.merge(out_channels)


def autocrop(image, threshold=15):

    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2

    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        image = image[:1, :1]

    return image


def autobrightnessandcontrast(image):

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    channels = cv2.split(hsv_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32, 32))
    channels[2] = clahe.apply(channels[2])
    channels[2] = cv2.bilateralFilter(channels[2], 5, 75, 75)
    hsv_image = cv2.merge(channels)
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


def rotateImage_colour(image, angle):
  i = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image_center = tuple(np.array(i.shape)/2)
  rot_mat = cv2.getRotationMatrix2D(center=image_center,angle=angle,scale=1.0)
  result = cv2.warpAffine(image, rot_mat, i.shape,flags=cv2.INTER_LINEAR)
  return result

def preprocess_image(file, train_or_test, output_path):
    if file.endswith('.jpeg') & (os.path.getsize(input_path + file) > 0):
        image = cv2.imread(input_path + file)
        img = autocrop(image)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        img = simplest_cb(img, 1)

        # TODO : check if image requires brightness and contrast adjustment
        img = autobrightnessandcontrast(img)

        cv2.imwrite(output_path + file, img)

        if train_or_test == 'train':
            flipped_img = cv2.flip(img, 1)

            rot_45 = rotateImage_colour(img, 45)
            rot_90 = rotateImage_colour(img, 90)
            rot_270 = rotateImage_colour(img, 270)

            rot_flipped_45 = rotateImage_colour(flipped_img, 45)
            rot_flipped_90 = rotateImage_colour(flipped_img, 90)
            rot_flipped_270 = rotateImage_colour(flipped_img, 270)

            cv2.imwrite(output_path + 'rot45_' + file, rot_45)
            cv2.imwrite(output_path + 'rot90_' + file, rot_90)
            cv2.imwrite(output_path + 'rot270_' + file, rot_270)

            cv2.imwrite(output_path + 'flipped_' + file, flipped_img)
            cv2.imwrite(output_path + 'flipped_rot45_' + file, rot_flipped_45)
            cv2.imwrite(output_path + 'flipped_rot90_' + file, rot_flipped_90)
            cv2.imwrite(output_path + 'flipped_rot270_' + file, rot_flipped_270)
#-------------------------------------------#


def process(ipath, tt, opath):

    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(preprocess_image)(file, tt, opath) for file in os.listdir(ipath))


if __name__ == "__main__":

    opts, args = getopt.getopt(sys.argv[1:], 'i:o:t:', ["ipath=","opath=","tr_te"])

    for opt, arg in opts:
        if opt in ("-i", "--ipath"):
            input_path = arg
        elif opt in ("-o", "--ofile"):
            output_path = arg
            if not os.path.exists(os.path.dirname(output_path)):
                try:
                    os.makedirs(os.path.dirname(output_path))
                except OSError as exc:  # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
        elif opt in ("-t", "--tr_te"):
            train_or_test = arg

    process(input_path, train_or_test, output_path)