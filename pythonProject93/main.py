import numpy as np
import cv2
import os
import face_recognition

data = {'Студент':[],'Преподаватель':[],'Технический специалист':[]}

with open('Persons') as r:
    for i in r.read().split('\n'):
        if i in data.keys():
            key = i
        else:
            data[key].append(i)

path = 'KnownFaces'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cls in myList:
    absol = os.path.abspath(f'{path}//{cls}')
    curImg = cv2.imread(absol)
    images.append(curImg) #читаем изображение, сохраняем результат в список
    classNames.append(os.path.splitext(cls)[0])

print(classNames)

def findEncodings(images): #декодирование
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #изменение цвета
        encode = face_recognition.face_encodings(img)[0] #распознавание лица
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images) #декодируем изображения
print("Декодирование закончено")

cap = cv2.VideoCapture(0) #подключение к камере

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25) #картинка с камеры
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS) #поиск лиц
    encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame) #распознавание лиц

    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace) #сравнение лиц с БД
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace) #вычисление точности распознавания
        print("Точность распознавания")
        matchIndex = np.argmin(faceDis) #индекс наименьшей вероятности несовпадения
        print(100-faceDis[matchIndex]*100, "%")


        name = "Неизвестный пользователь"
        if 1-faceDis[matchIndex]>=0.6:
            name = classNames[matchIndex] #имя
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            #дальше идёт проверка статуса
            if name in data['Студент']:
                status = 'Студент'
                color = (153,255,153)
            elif name in data['Преподаватель']:
                status = 'Преподаватель'
                color = (225, 0, 0)
            else:
                status = 'Технический специалист'
                color = (0, 0, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
            cv2.putText(img, f'{name} ({status})', (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            print(f'{name} ({status})')
        else:
            print("нет доступа")
            color = (0, 77, 255)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
            cv2.putText(img, f'{name}', (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)





    cv2.imshow("WebCam", img)
    cv2.waitKey(1)
