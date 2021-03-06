import numpy as np

def get_box_dimensions(outputs, height, width):
    boxes =[]
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:] #   lưu trữ độ tin cậy tương ứng với từng đối tượng.
            print(scores)
            class_id = np.argmax(scores) 
            conf = scores[class_id] # xác định chỉ số của lớp có độ tin cậy
            if conf > 0.3 :
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x,y,w,h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids