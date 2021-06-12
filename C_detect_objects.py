'''
    - Để dụ đoán chính xác các đối tượng có mạng deep learning, chúng rta cần xử lí dữ liệu 
    của mình mà module cv2.dnn cung cấp cho chúng ta 2 chức năng cho mục đích này là: 
        + blobFromImage
        + blobFromImage. 
    - Các chức năng này thực hiện chia tỉ lệ, trừ trung bình và hoán đổi các channels tùy chọn 
'''
import cv2
from A_load_yolo import *
from B_load_image import *
def _detect_objects(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(img, 
                                 scalefactor = 0.00392, # =(1/255) chia tỷ lệ pixel hình ảnh trong phạm vi từ 0 đến 1
                                 size = (320,320), 
                                 mean = (0,0,0), # Không cần phép trừ trung bình
                                 swapRB=True, 
                                 crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers) #rả về một danh sách lồng nhau chứa 
    #thông tin về tất cả các đối tượng được phát hiện bao gồm tọa độ (x, y) 
    #của tâm đối tượng được phát hiện, (w, h) của bbox, be called offsets
    #độ tin cậy và điểm cho tất cả các lớp đối tượng được liệt kê trong coco.names.
    return blob, outputs