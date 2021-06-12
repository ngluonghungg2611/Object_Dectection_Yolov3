'''
    Bây giờ chúng ta đã có các đỉnh của hộp giới hạn dự đoán và class_id 
    (chỉ mục của lớp đối tượng được dự đoán), chúng ta cần vẽ hộp giới hạn 
    và thêm nhãn đối tượng vào nó. Bây giờ chúng ta đã có các đỉnh của hộp
    giới hạn dự đoán và class_id (chỉ mục của lớp đối tượng được dự đoán), 
    chúng ta cần vẽ hộp giới hạn và thêm nhãn đối tượng vào nó. 
'''
import cv2
from D_get_box_dimensions import*

def draw_labels(boxes, confs, colors, class_ids, classes, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label, (x,y - 5), font, 1, color, 1)
    cv2.imshow("image", img)