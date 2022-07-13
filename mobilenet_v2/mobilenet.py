
import cv2
import tensorflow as tf
import numpy as np
import time   # for basic timing of inference

classes = open('classes.txt').read().splitlines()

font = cv2.FONT_HERSHEY_SIMPLEX

img = cv2.imread('../images/kitchen.jpg')
img_height, img_width, img_colors = img.shape
print('img.shape',img.shape)
cv2.imshow('input image',img)

img_tensor = np.array([img])
print('img_tensor.shape',img_tensor.shape)

print()
print("Tensorflow will complain below as the model version is old, and if there is no CUDA.")
print("These warning messages can be ignored.")
print()

t = time.perf_counter()
model = tf.keras.models.load_model('ssd_mobilenet_v2_fpnlite_640x640_1')
d = time.perf_counter() - t

print(f"MobileNet V2 trained model loaded in {d:.3f} seconds.")

t = time.perf_counter()
output = model(img_tensor)
d = time.perf_counter() - t

print(f"Inference pass completed in {d:.3f} seconds.")

boxes = output["detection_boxes"][0]

output_img = img.copy()

for i in range(10):
    box = boxes[i] # [ymin, xmin, ymax, xmax]

    top_left = (int(box[1] * img_width), int(box[0] * img_height))
    bottom_right = (int(box[3] * img_width), int(box[2] * img_height))
    #                                                         B  G R  thickness
    output_img = cv2.rectangle(output_img, top_left, bottom_right, (255,0,0), 2)

    class_index = int(output["detection_classes"][0][i])
    class_label = classes[class_index-1]
    text_top_left = (int(box[1] * img_width), int(box[0] * img_height - 2))

    detection_score = output["detection_scores"][0][i]

    box_text = f"{class_label} {detection_score:.2f}" # E.g. "person 0.79"

    #                                                             scale   B G R
    output_img = cv2.putText(output_img, box_text, text_top_left, font, 0.7, (255,0,0), 1, cv2.LINE_AA)

cv2.imshow('output',output_img)

print("Select an image window and hit any key to complete")
cv2.waitKey(0)
cv2.destroyAllWindows()
