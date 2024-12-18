from PIL import Image
from numpy import asarray
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
import os
import cv2


def rename_files(img, label):
    for filename in os.listdir(img):
        if filename.endswith(".png"):
            new_name = filename.replace('Img', '')
            old_file = os.path.join(img, filename)
            new_file = os.path.join(img, new_name)
            os.rename(old_file, new_file)

    for filename in os.listdir(label):
        if filename.endswith(".png"):
            new_name = filename.replace('label', '')
            old_file = os.path.join(label, filename)
            new_file = os.path.join(label, new_name)
            os.rename(old_file, new_file)        

# img_dir = '/home/darcy/PycharmProjects/PMSD-Tracking/SeqTrack/data/custom/aug/20/img'
# label_dir = '/home/darcy/PycharmProjects/PMSD-Tracking/SeqTrack/data/custom/aug/20/label'
# rename_files(img_dir, label_dir)

def reorder_filename(img, label):
    frame_list = [frame for frame in os.listdir(img) if frame.endswith(".png")]
    frame_list.sort(key=lambda f: int(f[:-4]))
    count = 0 
    for frame in frame_list:
        old_file = os.path.join(img, frame)
        name = '{:04}'.format(count) + '.png'
        img_name  = os.path.join(img, name)
        count += 1
        os.rename(old_file, img_name)

# img_dir = '/home/darcy/PycharmProjects/PMSD-Tracking/SeqTrack/data/custom/val/1/img'
# label_dir = '/home/darcy/PycharmProjects/PMSD-Tracking/SeqTrack/data/custom/val/1/label'
# reorder_filename(img_dir, label_dir)

def resize_image(img_dir, label_dir):
    for filename in os.listdir(img_dir):
        if filename.endswith(".png"):
            image = Image.open(os.path.join(img_dir, filename))
            new_image = image.resize((356, 427))
            new_image.save(os.path.join(img_dir, filename))

    for filename in os.listdir(label_dir):
        if filename.endswith(".png"):
            image = Image.open(os.path.join(label_dir, filename))
            new_image = image.resize((356, 427))
            new_image.save(os.path.join(label_dir, filename))      

# img_dir = '/home/darcy/PycharmProjects/PMSD-Tracking/SeqTrack/data/custom/aug/16/img'
# label_dir = '/home/darcy/PycharmProjects/PMSD-Tracking/SeqTrack/data/custom/aug/16/label'
# resize_image(img_dir, label_dir)

def draw_circle_bounding_box(img_dir, label_dir):
    frame_list = [frame for frame in os.listdir(label_dir) if frame.endswith(".png")]
    frame_list.sort(key=lambda f: int(f[:-4]))
    label_text_dir = os.path.join(label_dir, 'label.txt')
    label_text = open(label_text_dir, 'w')
    for frame in frame_list:
        img = os.path.join(label_dir, frame)
        # Label connected components in the binary image
        image = asarray(Image.open(img))
        labeled_img = label(image)
        # Extract properties of the labeled regions (in this case, the circle)
        regions = regionprops(labeled_img)
        # Get the bounding box coordinates of the circle
        minr, minc, maxr, maxc = [0, 0, 0, 0]
        if regions:
            minr, minc, maxr, maxc = regions[0].bbox
        label_text.writelines([str(max(0, minc)), ',', str(max(0, minr)), ',',
                               str(maxc - minc), ',', str(maxr - minr), '\n'])
        # else: 
        #     os.remove(os.path.join(img_dir, frame));                       
    label_text.close()


# label_dir = '/home/darcy/PycharmProjects/PMSD-Tracking/SeqTrack/data/custom/train/14/label'
# img_dir = '/home/darcy/PycharmProjects/PMSD-Tracking/SeqTrack/data/custom/train/14/img'
# draw_circle_bounding_box(img_dir, label_dir)


def draw_bounding_box(directory, labels):
    frame_list = [frame for frame in os.listdir(directory) if frame.endswith(".png")]
    frame_list.sort(key=lambda f: int(f[:-4]))
    lable_file = open(labels, "r")
    for frame in frame_list:
        box = lable_file.readline().split(',')
        image_path = os.path.join(directory, frame)
        image = asarray(Image.open(image_path))

        # Plot the original image
        plt.imshow(image, cmap='gray')

        # Draw bounding box around the circle
        rect = plt.Rectangle((int(box[0]), int(box[1])), int(box[2]), int(box[3]), fill=False, edgecolor='red',
                             linewidth=2)
        plt.gca().add_patch(rect)

        plt.title('Bounding Box for ' + frame)
        plt.show()
        break


# dir = '/home/darcy/PycharmProjects/PMSD-Tracking/SeqTrack/data/custom/test/1/img'
# labels = '/home/darcy/PycharmProjects/PMSD-Tracking/SeqTrack/data/custom/test/1/label/label.txt'
# draw_bounding_box(dir, labels)

def save_labeled_image(directory, labels, save_dir):
    i = 0
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    frame_list = [frame for frame in os.listdir(directory) if frame.endswith(".png")]
    frame_list.sort(key=lambda f: int(f[:-4]))
    label_file = open(labels, "r")
    for frame in frame_list:
        box = label_file.readline().split()
        image_path = os.path.join(directory, frame)
        image = cv2.imread(image_path)
        start_point = (int(box[0]), int(box[1]))
        end_point = (int(box[0]) + int(box[2]), int(box[1]) + int(box[3]))
        color = (255, 0, 0)
        labeled_image = Image.fromarray(cv2.rectangle(image, start_point, end_point, color, thickness=2), 'RGB')
        # filename = str(i) + '.png'
        img = labeled_image.save(os.path.join(save_dir, frame))
        # i += 1
    print("Done")


dir = '/home/darcy/PMSD-Tracking/SeqTrack/data/custom/test/5/img'
save_dir = '/home/darcy/PMSD-Tracking/SeqTrack/test/tracking_results/seqtrack/seqtrack_b256/custom/5'
labels = '/home/darcy/PMSD-Tracking/SeqTrack/test/tracking_results/seqtrack/seqtrack_b256/custom/5.txt'

save_labeled_image(dir, labels, save_dir)

def calculate_IoU(prediction, ground_truth):
    pred = open(prediction, "r")
    truth = open(ground_truth, "r")
    iou = []

    while True:
        pred_box = pred.readline().split()
        truth_box = truth.readline().strip().split(',')
        if not pred_box:
            break
        if int(truth_box[2]) == 0 or int(truth_box[3] == 0):
            iou.append(0)
            continue

        # determine the coordinates of the intersection rectangle
        x_left = max(int(pred_box[0]), int(truth_box[0]))
        y_top = max(int(pred_box[1]), int(truth_box[1]))
        x_right = min(int(pred_box[0]) + int(pred_box[2]), int(truth_box[0]) + int(truth_box[2]))
        y_bottom = min(int(pred_box[1]) + int(pred_box[3]), int(truth_box[1]) + int(truth_box[3]))

        if x_right < x_left or y_bottom < y_top:
            iou.append(0.0)
            continue

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        pred_area = int(pred_box[2]) * int(pred_box[3])
        truth_area = int(truth_box[2]) * int(truth_box[3])
        iou.append(intersection_area / float(pred_area + truth_area - intersection_area))
        # iou.append(intersection_area / float(truth_area))

    pred.close()
    truth.close()

    IoU = np.array(iou)
    np.savetxt('/home/darcy/PMSD-Tracking/SeqTrack/test/tracking_results/seqtrack/seqtrack_b256/custom/5/IoU.txt',
               IoU, fmt="%f")

    plt.plot(IoU)
    title = "Mean = " + str(np.mean(IoU[IoU > 0])) + ", STD = " + str(np.std(IoU[IoU > 0]))
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylim(0.0, 1.0)
    plt.ylabel("IoU")
    plt.savefig('/home/darcy/PMSD-Tracking/SeqTrack/test/tracking_results/seqtrack/seqtrack_b256/custom/5/IoU.png')
    # b = np.loadtxt('test1.txt', dtype=int)


prediction = '/home/darcy/PMSD-Tracking/SeqTrack/test/tracking_results/seqtrack/seqtrack_b256/custom/5.txt'
ground_truth = '/home/darcy/PMSD-Tracking/SeqTrack/data/custom/test/5/label/label.txt'
calculate_IoU(prediction, ground_truth)
 

def convert_masks(img_dir, plague_dir, save_dir):
    frame_list = [frame for frame in os.listdir(img_dir) if frame.endswith(".png")]
    frame_list.sort(key=lambda f: int(f[:-4]))
    for frame in frame_list:
        img = os.path.join(plague_dir, frame)
        image = asarray(Image.open(img))
        image = image * 255
        labeled_img = Image.fromarray(image)
        labeled_img.save(os.path.join(save_dir, frame))
        break



# img_dir = '/home/darcy/PycharmProjects/PMSD-Tracking/SeqTrack/data/custom/test/5/img'
# mask_dir = '/home/darcy/PycharmProjects/PMSD-Tracking/SeqTrack/data/custom/test/5/mask'
# artery_dir = '/home/darcy/PycharmProjects/PMSD-Tracking/SeqTrack/data/custom/test/5/label'

# convert_masks(img_dir, artery_dir, mask_dir)