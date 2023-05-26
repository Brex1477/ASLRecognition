import pickle
import cv2
import mediapipe as mp
import numpy as np
import mouse

#Loading pickle data
model_dict = pickle.load(open('.\model.p', 'rb'))
model = model_dict['model']

#Instantiating camera
cap = cv2.VideoCapture(0)

#Mediapipe solutions
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
maxNumHands = 1
hands = mp_hands.Hands(max_num_hands=maxNumHands, min_detection_confidence=0.7)

#Control Mouse Toggle
cntrlMouse = False

#Lable List
labels_dict = {0: 'Fuck You', 1: 'Hand'}

#Body
while cap.isOpened():
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    #Get Window Size
    H, W, _ = frame.shape

    #Image correction
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image, 1)
    image.flags.writeable = False
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            #Rect controls
            x_max = 0
            y_max = 0
            x_min = W
            y_min = H

            #Draw Hand Points
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for lm in hand_landmarks.landmark:
                #Rect Size calculator
                x, y = int(lm.x * W), int(lm.y * H)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y

                #X,Y lists
                x_.append(lm.x)
                y_.append(lm.y)
            

            #Building readable predict data
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

        #Predict Gesture
        prediction = model.predict([np.asarray(data_aux)])

        #Get label
        predicted_character = labels_dict[int(prediction[0])]

        #Mouse control
        if cntrlMouse:
            mouse.move(x_min*5, y_min*5)
            if predicted_character == "Fuck You":
               mouse.click('left') 
        else:
            #Rectangle in text Builder
            cv2.rectangle(image, (x_min-10, y_min-10), (x_max+10, y_max+10), (0, 255, 0), 2)
            cv2.putText(image, predicted_character, (x_min - 10, y_min - 20), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 3,cv2.LINE_AA)

    #Window Essentials
    cv2.imshow('ASL Recognition', image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    