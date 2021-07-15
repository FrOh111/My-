import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'Person'
images = []
classNames = []
verifyFaces = [] #用于存储每次检测出来的人脸，如果勿检了就会和上一个不一样，当连续n个脸都相同则认为检测成功
verifyTime = 0
mylist = os.listdir(path)
# print(mylist)

#把图片文件存到images中，把文件的人名存到classNames中
for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

#把人脸的encoding存到列表encodeList中
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        encodeList.append(face_recognition.face_encodings(img)[0])
    return encodeList

#记录人脸出现时间
def appearTime(name):
    with open('AppearTime.csv','r+') as f:
        #为了使重复出现的人脸不进行二次读入，故使用list,先逐行读取文件中已存在的人脸
        #每个出现的人脸都放到nameList中，这样做只有在出现新人脸时列表才会变化
        appearList = f.readlines()
        nameList = []
        for line in appearList:
            appear = line.split(',')
            nameList.append(appear[0])
        #出现新的人脸，把脸的数据和出现时间存到数据库
        if name not in nameList:
            now = datetime.now()
            timeStr = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{timeStr}')
            f.close()


#录入到系统中的人脸参数
encodeFaceKnown = findEncodings(images)
print(classNames)
print('读入库中人脸参数完成')

#从摄像头中检测当前库中存在的人脸
cap = cv2.VideoCapture(0)
while True:
    success,img = cap.read()
    #减小图像，提高读取效率
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)

    imgBGR = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faceLocs = face_recognition.face_locations(imgBGR)
    encode_faces = face_recognition.face_encodings(imgBGR,faceLocs)
    # cv2.rectangle(imgS, (faceLocs[3], faceLocs[0]), (faceLocs[1], faceLocs[2]), (255, 0, 0), 2)

    for encode_face,faceLoc in zip(encode_faces,faceLocs):  #这样写就在一个numpy中显示了
        matchs = face_recognition.compare_faces(encodeFaceKnown,encode_face)
        faceDis = face_recognition.face_distance(encodeFaceKnown,encode_face)
        # print(type(faceDis))
        matchIndex = np.argmin(faceDis)
        # print(matchIndex)


        if matchs[matchIndex]:
            # print(classNames[matchIndex])
            #接下来把小图像中检测出来的画面输出到大图像上
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            # cv2.rectangle(img, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 0), 2)
            cv2.rectangle(img,(x1,y1), (x2,y2), (255, 0, 0), 2)
            cv2.putText(img,classNames[matchIndex],(x1,y1+25),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),2)

            if len(verifyFaces) == 0:
                verifyFaces.append(matchIndex)
            if matchIndex != verifyFaces[0]:
                # print("换了一个人", matchIndex)
                verifyFaces.clear()
                verifyTime = 0
            else:
                verifyTime += 20
                #记录到文件中
                if verifyTime >= 100:
                    verifyTime = 100
                    cv2.putText(img, "success", (x1 + 200, y1 + 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
                    appearTime(classNames[matchIndex])
                    print("录入成功：",classNames[matchIndex])
            cv2.putText(img, str(verifyTime), (x1 + 200, y1 + 25), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Output',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break