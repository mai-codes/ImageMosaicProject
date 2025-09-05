import sys
import numpy as np
from imageio import imread
from imageio import imsave
from skimage import img_as_ubyte
from create_panorama import process_images
from lucas_kanade import bilinear_interp

def read_camera_parameters():
    with open('camera_values.txt', 'r') as f:
        focal_length = float(f.readline().strip())
        k1 = float(f.readline().strip())
        k2 = float(f.readline().strip())
    return focal_length, k1, k2

def reproject_to_cylindrical(imagefiles):
    cylinder_images = []
    for file in imagefiles:
        cylinder_file = file.replace("." + file.split(".")[-1], "_cyl" + "." + file.split(".")[-1])

        print("Reprojecting '%s' from planar to cylindrical coordinates ... " % (file))
        img = imread(file).astype(np.float32)/255.
        cyl = cylinder_calculations(img)
        imsave(cylinder_file, img_as_ubyte(cyl))
        cylinder_images.append(cylinder_file)
    return cylinder_images

def cylinder_calculations(image):
    focal_length, k1, k2 = read_camera_parameters()
    cylindrical_points = np.mgrid[:image.shape[0], :image.shape[1]].transpose(1,2,0).astype(np.float32)
    x_cyl = cylindrical_points[:, :, 1]
    y_cyl = cylindrical_points[:, :, 0]
 
    theta = (x_cyl - image.shape[1] / 2.) / focal_length
    h = (y_cyl - image.shape[0] / 2.) / focal_length

    x_hat = np.sin(theta)
    y_hat = h
    z_hat = np.cos(theta)

    r2 = np.power(x_hat, 2) + np.power(y_hat, 2)
    distort_factor = 1 + k1 * r2 + k2 * np.power(r2, 2)
    x_hat_prime = x_hat/distort_factor
    y_hat_prime = y_hat/distort_factor

    x = focal_length * (x_hat_prime/z_hat) + image.shape[1] / 2.
    y = focal_length * (y_hat_prime/z_hat) + image.shape[0] / 2.

    cylindrical_points[:, :, 1] = x
    cylindrical_points[:, :, 0] = y

    image_cyl = bilinear_interp(image, cylindrical_points)
    return image_cyl

if __name__ == "__main__":
    imagefiles = []
    x_init_disps = []
    for file in open(sys.argv[1]).readlines():
        imagefiles.append(file.strip().split()[0])

    for file in open(sys.argv[2]).readlines():
        x_init_disps.append(file.strip().split()[0])

    cylindrical_images = reproject_to_cylindrical(imagefiles)

    xvals = np.array(x_init_disps)[:, np.newaxis]
    yvals = np.zeros((len(cylindrical_images)))[:, np.newaxis]
    x_y_displacements = np.hstack([xvals, yvals])

    images = []
    for img in cylindrical_images:
        images.append(imread(img)[:,:,:3].astype(np.float32)/255.)

    panorama = process_images(images, x_y_displacements)
    print("Saving the panorama...")
    imsave("panorama_images/panorama_result.png", img_as_ubyte(panorama))
