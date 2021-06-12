from A_load_yolo import _load_yolo
from B_load_image import _load_image
from C_detect_objects import _detect_objects
from D_get_box_dimensions import get_box_dimensions
from E_draw_labels import draw_labels
import cv2
def image_detect(img_path):
    model, classes, colors, output_layers = _load_yolo()
    image, height, width, channels = _load_image(img_path)
    blob, outputs = _detect_objects(image, model, output_layers)
    boxes, confs, class_ids, = get_box_dimensions(outputs, height, width)
    draw_labels(boxes, confs, colors, class_ids, classes, image)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'): # Nhấn phím q để dừng
            break