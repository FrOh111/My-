import cv2
import numpy as np
from datetime import datetime

threshold = 0.6
nms_threshold = 0.2

cap = cv2.VideoCapture(0)

classNames = []
classFile = 'Resources/coco.names'

with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
#print(classNames)

configPath = 'Resources/ssd_mobilenet_v3_large_coco.pbtxt'
weightsPath = 'Resources/frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath,configPath)

net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

# 定义编解码器并创建VideoWriter对象
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')    #输出MP4文件
fps = cap.get(cv2.CAP_PROP_FPS)
save_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
save_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
start_time = datetime.now()
out = cv2.VideoWriter('{time}.avi'.format(time=start_time.strftime('%y年%m月%d日%H点%M分%S开始')),fourcc,10,(save_width,save_height))   # 视频名字为今天的日期和开始时分

while True:
    success,img = cap.read()
    classIds,confs,bbox = net.detect(img,confThreshold=threshold)

    #indices = cv2.dnn.NMSBoxes(bbox,confs,threshold,nms_threshold=nms_threshold)    #需要把bbox和confs转变成list
    #bbox原来是numpy.ndarray类型，诸如[[336 154 180 323]]
    #confs原来也是numpy.ndarray类型

    bbox = list(bbox)
    #   注：把confs强转成np.array是为了使用reshape方法
    confs = list(np.array(confs).reshape(1,-1)[0])    #这时的confs中每一项属于numpy.float32类型，需要用map转成float类型
    confs = list(map(float,confs))

    indices = cv2.dnn.NMSBoxes(bbox, confs, threshold, nms_threshold=nms_threshold)
    #   此时的indices返回bbox的索引值

    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img, (x,y),(x+w,y+h), color=(255, 0, 0), thickness=2)
        cv2.putText(img, classNames[classIds[i][0] - 1], (box[0], box[1] + 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
        cv2.putText(img, str(round(confs[0] * 100, 1)), (box[0] + 150, box[1] + 30), cv2.FONT_HERSHEY_DUPLEX, 1,
                    (255, 0, 0), 2)
        if classNames[classIds[i][0] - 1] == 'person':
            success, imgPerson = cap.read()
            now = datetime.now()
            timeStr = now.strftime('%H:%M:%S')
            # print(timeStr)
            cv2.putText(imgPerson, str(timeStr), (450, 50), cv2.FONT_HERSHEY_DUPLEX, 1,
                        (255, 0, 0), 2)
            cv2.imshow('People', imgPerson)
            out.write(imgPerson)
            # cv2.destroyWindow('People')

    cv2.imshow('Output',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break