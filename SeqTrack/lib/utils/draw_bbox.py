from PIL import Image
from numpy import asarray
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
import os
import cv2


def rename_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            new_name = filename.replace('DUPLEX STUDIE  EINS^^^^', '')
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_name)
            os.rename(old_file, new_file)

# dir = '/home/darcy/PycharmProjects/PMSD-Tracking/SeqTrack/data/got10k/test/1'
# diseased = '/home/darcy/PycharmProjects/PMSD-Tracking/SeqTrack/data/diseased'
# rename_files(diseased)


def draw_circle_bounding_box(image, labeled):
    # Label connected components in the binary image
    print(type(labeled))
    labeled_img = label(labeled)

    # Extract properties of the labeled regions (in this case, the circle)
    regions = regionprops(labeled_img)

    if len(regions) == 0:
        print("No circle found.")
        return

    # Get the bounding box coordinates of the circle
    minr, minc, maxr, maxc = regions[0].bbox

    # Plot the original image
    plt.imshow(image, cmap='gray')

    # Draw bounding box around the circle
    # rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
    rect = plt.Rectangle((220.0000,194.0000), 115.0000, 117.0000, fill=False, edgecolor='red', linewidth=2)
    plt.gca().add_patch(rect)

    plt.title('Bounding Box Around the Circle')
    plt.show()


# image = Image.open('/home/darcy/PMSD/continuous/carotid/1/img/Img0100.png')
# lab = Image.open('/home/darcy/PMSD/continuous/carotid/1/label/label0100.png')
# labeled = asarray(lab)
# diseased = Image.open('/home/darcy/PycharmProjects/PMSD-Tracking/SeqTrack/data/got10k/test/1/089.png')
# # whale = Image.open('/home/darcy/PycharmProjects/VideoX/SeqTrack/data/got10k/test/GOT-10k_Test_000001/00000001.jpg')
#
# draw_circle_bounding_box(diseased, labeled)

def draw_bounding_box(directory, labels):
    frame_list = [frame for frame in os.listdir(directory) if frame.endswith(".png")]
    frame_list.sort(key=lambda f: int(f[:-4]))
    lable_file = open(labels, "r")
    for frame in frame_list:
        box = lable_file.readline().split()
        image_path = os.path.join(directory, frame)
        image = asarray(Image.open(image_path))

        # Plot the original image
        plt.imshow(image, cmap='gray')

        # Draw bounding box around the circle
        rect = plt.Rectangle((int(box[0]), int(box[1])), int(box[2]), int(box[3]), fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)

        plt.title('Bounding Box for ' + frame)
        plt.show()


# dir = '/home/darcy/PycharmProjects/PMSD-Tracking/SeqTrack/data/got10k/test/1'
# labels = '/home/darcy/PycharmProjects/PMSD-Tracking/SeqTrack/test/tracking_results/seqtrack/seqtrack_b256_got/got10k/1.txt'
#
# draw_bounding_box(dir, labels)

def save_labeled_image(directory, labels, save_dir): 
    frame_list = [frame for frame in os.listdir(directory) if frame.endswith(".png")]
    frame_list.sort(key=lambda f: int(f[:-4]))
    lable_file = open(labels, "r")
    for frame in frame_list:
        box = lable_file.readline().split()
        image_path = os.path.join(directory, frame)
        image = cv2.imread(image_path)
        start_point = (int(box[0]), int(box[1]))
        end_point = (int(box[0]) + int(box[2]), int(box[1]) + int(box[3]))
        color = (255, 0, 0)
        labeled_image = Image.fromarray(cv2.rectangle(image, start_point, end_point, color, thickness=2), 'RGB')
        img = labeled_image.save(os.path.join(save_dir, frame))
    print("Done")


dir = '/home/darcy/PycharmProjects/PMSD-Tracking/SeqTrack/data/got10k/test/1'
save_dir = '/home/darcy/PycharmProjects/PMSD-Tracking/SeqTrack/test/tracking_results/seqtrack/seqtrack_b256_got/got10k/labeled'
labels = '/home/darcy/PycharmProjects/PMSD-Tracking/SeqTrack/test/tracking_results/seqtrack/seqtrack_b256_got/got10k/1.txt'

save_labeled_image(dir, labels, save_dir)