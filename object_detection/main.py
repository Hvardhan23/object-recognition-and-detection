import cv2 

cap = cv2.VideoCapture(0) # using laptop/usb cam for live vid
cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)

class_names= []
class_file= 'coco.names'
with open(class_file,'rt') as f:
    class_names= f.read().rstrip('\n').rsplit('\n')

print(class_names) #getting names file into list

config_path= 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weight_path= 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weight_path,config_path)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


while True: # for cont vid
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=0.5) # more than 50% confidence
    print(classIds,bbox)

    if len(classIds) != 0: #when it detects smtg from list
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color=(0,255,0),thickness=2)
            cv2.putText(img,class_names[classId-1].upper(),(box[0]+10,box[1]+10),
            cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),2)
            cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
            cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    cv2.imshow('Output',img)
    cv2.waitKey(1)