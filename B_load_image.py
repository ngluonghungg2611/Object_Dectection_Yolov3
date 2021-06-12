'''
    Để chấp nhận hình ảnh, chúng ta sẽ cần 1 hàm khác gọi là hàm _load_image().
    Hàm này sẽ chấp nhận 1 đường dẫn hình ảnh (img_path) làm tham số, đọc hình ảnh, thay đổi kích thước và trả về
'''
import cv2
from A_load_yolo import*
def _load_image(img_path):
    """
    intput:
        img_paht: link dan hinh anh
    output:
        img: hinh anh
        height, widht: kich thuong hinh anh
        channels: lop
    """
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    return img, height, width, channels
    