import time
import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)  # Подключение к web-камере 
mp_Hands = mp.solutions.hands  # говорим, что хотим распознать руки
hands = mp_Hands.Hands(max_num_hands = 2)  # характеристики для распознавания
mpDraw = mp.solutions.drawing_utils  # инициализация утилит рисования

fingers_coord = [(8, 6), (12, 10), (16, 14), (20, 18)]  # координаты ключевых точек на руке, кроме большого пальца
thumb_coord = (4, 3) # координаты ключевых точек для большого пальца

while cap.isOpened():  # пока камера "работает"
    success, image = cap.read()  # получаем кадр с камеры
    if not success: # если не удалось получить кадр с камеры
        print('Не удалось получить кадр с web-камеры')
        continue  # переход к ближайшему циклу (while 12 строчка)
    prevTime = time.time() 
    image = cv2.flip(image, 1)   # зеркально отражаем изображение
    RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  #BGR -> RGB изображение
    results = hands.process(RGB_image)  # ищем руки на изображении
    multiLandMarks = results.multi_hand_landmarks  # Извлекаем список найденных рук
    
    if multiLandMarks: # если руки найдены
        upCount = 0
        for idx, handLms in enumerate(multiLandMarks): # перебираем найденные руки
            lbl = results.multi_handedness[idx].classification[0].label
            print(lbl)

            mpDraw.draw_landmarks(image, handLms, mp_Hands.HAND_CONNECTIONS)
            fingersList = [] 
            
            for lm in handLms.landmark: # перебираем ключевые точки на руке
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                fingersList.append((cx, cy))
            for coordinate in fingers_coord:
                if fingersList[coordinate[0]][1] < fingersList[coordinate[1]][1]:
                    upCount += 1
            
            side = 'left'
            if fingersList[5][0] > fingersList[17][0]:
                side = 'right'
            if side == 'left':
                if fingersList[thumb_coord[0]][0] < fingersList[thumb_coord[1]][0]:
                    upCount += 1 
            else:
                if fingersList[thumb_coord[0]][0] > fingersList[thumb_coord[1]][0]:
                    upCount += 1     
        cv2.putText(image, str(upCount), (50, 100), cv2.FONT_HERSHEY_DUPLEX, 5, (50, 200, 255), 5)            
        print(upCount)

    currentTime = time.time()
    fps = 1 / (currentTime - prevTime)
    cv2.putText(image, f"FPS: {fps}", (150, 100), cv2.FONT_HERSHEY_DUPLEX, 3, (200, 200, 200), 3 )
    cv2.imshow('web-cam', image)
    if cv2.waitKey(1) & 0xFF == 27: # ожидаем нажание клавиши ESC
        break

cap.release()