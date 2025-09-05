import numpy as np
from lucas_kanade import pyramid_lucas_kanade, translate

BLEND_WINDOW = 512
LEVELS = 4
STEPS = 5

def process_images(images, initial_displacements):
    final_displacements = calculate_final_displacements(images, initial_displacements)

    x_displacements = np.transpose(final_displacements)[0]
    x_sum = 0
    for x in x_displacements:
        x_sum += x

    y_displacements = np.transpose(final_displacements)[1]
    y_sum, y_max = 0, 0
    for y in y_displacements:
        y_sum += y
        y_max = np.maximum(y_sum, y_max)

    height = images[0].shape[0]
    width = images[0].shape[1]
    pano_width = width + int(np.abs(x_sum))
    pano_height = height + int(np.abs((y_max)))
    shape = [pano_height, pano_width, 3]
    
    x_initial = x_displacements[-1]
    y_initial = y_displacements[-1]
    init_pos = [x_initial, y_initial + y_max]

    print("Making panorama...")
    panorama = generate_panorama(images, shape, final_displacements, init_pos)
    return panorama

def generate_panorama(images, shape, displacements, initial_position):
    panorama, panorama_mask, new_panorama = np.empty((shape)), np.empty((shape)), np.empty((shape))

    panorama, height, width, x_displaced, y_displaced = locate_image_in_panorama(images[-1], panorama, initial_position, displacements)
    
    blending_mask = create_mask(height, width)
 
    i = 0
    while i < len(images):
        print("Stitching image %i of %i..." % (i + 1, len(images) + 1))
        current_img = crop_image(images[i])
        panorama = linear_blend(panorama, panorama_mask, new_panorama, blending_mask, height, width, x_displaced, y_displaced, current_img)

        x_displaced = x_displaced - displacements[i][0]
        y_displaced = y_displaced - displacements[i][1]
        i += 1
    return panorama

def locate_image_in_panorama(image, panorama, initial_position, displacements):
    located = crop_image(image)

    height = located.shape[0]
    width = located.shape[1]

    x_displaced = initial_position[0] - displacements[-1][0]
    y_displaced = initial_position[1] - displacements[-1][1]

    last_translated_img = translate(located,  -displacements[-1])
    y_pos = int(np.round(y_displaced))
    x_pos = int(np.round(x_displaced))
    panorama[y_pos:height + y_pos, x_pos:width + x_pos] = last_translated_img

    return panorama, height, width, x_displaced, y_displaced

def create_mask(height, width):
    mask = np.ones((height, width))
    i, j, = 0, 0 
    while i < height:
        mag = 0
        j = 0
        while j < width:
            if j <= BLEND_WINDOW:
                mask[i][j] = mag
                mag += 1/BLEND_WINDOW
            j+=1
        i+=1
    return mask

def crop_image(image):
    current_img = image
    current_height = current_img.shape[0]
    current_width = current_img.shape[1]
    new_width = current_width - BLEND_WINDOW
    current_img = current_img[0:current_height, BLEND_WINDOW:new_width]
    return current_img


def linear_blend(panorama, new_panorama, panorama_mask, mask, height, width, x_displaced, y_displaced, current_img):
    pm = mask[:, :, np.newaxis]
    y_pos = int(np.round(y_displaced))
    x_pos = int(np.round(x_displaced))

    panorama_mask[y_pos:height + y_pos, x_pos:width + x_pos] = pm
    new_panorama[y_pos:height + y_pos, x_pos:width + x_pos] = current_img

    result = (1 - panorama_mask/np.max(panorama_mask)) * panorama + panorama_mask/np.max(panorama_mask) * new_panorama
    return result

def calculate_final_displacements(images, initial_displacements):
    num_images = len(images)
    print("Using Pyramid LK to finalize displacements...")
    final_displacements = []
    for i in range(num_images):
        print("Calculating displacement %i of %i ..." % (i + 1, num_images))
        displacement = pyramid_lucas_kanade(images[i], images[(i+1) % num_images], initial_displacements[i], LEVELS, STEPS)
        final_displacements.append(displacement)

    return final_displacements