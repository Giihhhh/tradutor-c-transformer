import cv2
import mediapipe as mp
import numpy as np
import joblib

# carregar modelo
modelo = joblib.load("modelo_libras.pkl")

# inicializar mp
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# webcam
cap = cv2.VideoCapture(0)

while True:
   ret, frame = cap.read()
   if not ret:
       break

   # flip horizontal e conversão de cor
   frame = cv2.flip(frame, 1)
   rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

   # processar mãos
   resultados = hands.process(rgb)

   if resultados.multi_hand_landmarks:
       for mao in resultados.multi_hand_landmarks:
           # desenhar pontos na mão
           mp_drawing.draw_landmarks(frame, mao, mp_hands.HAND_CONNECTIONS)

           # extrair coordenadas normalizadas
           pontos = []
           for lm in mao.landmark:
               pontos.append(lm.x)
               pontos.append(lm.y)
               pontos.append(lm.z)

           # converter para np
           pontos = np.array(pontos).reshape(1, -1)

           # verificar tamanho do vetor ex: 21 pontos → 3 coordenadas = 63
           if pontos.shape[1] == 63:
               letra = modelo.predict(pontos)[0]
               prob = modelo.predict_proba(pontos).max()

               # mostrar resultado na tela
               cv2.putText(frame, f'{letra} ({prob:.0%})', (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                           2, (0, 255, 0), 3, cv2.LINE_AA)
               print(pontos)

   cv2.imshow("Reconhecimento de Libras", frame)

   # sair com a tecla 's'
   if cv2.waitKey(1) & 0xFF == ord('s'):
       break

cap.release()
cv2.destroyAllWindows()

