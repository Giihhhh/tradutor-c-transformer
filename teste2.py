import cv2
import mediapipe as mp
import numpy as np
import os
from glob import glob

video_path = input("Caminho do vídeo para debug (ex: videos/J_01.mp4): ").strip()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Erro ao abrir vídeo:", video_path)
    exit()

frame_idx = 0
found = False
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = hands.process(rgb)

    if resultados.multi_hand_landmarks:
        hand_landmarks = resultados.multi_hand_landmarks[0]
        # Imprime informação sobre a lista de landmarks
        print(f"Frame {frame_idx}: multi_hand_landmarks encontrado")
        print("type(hand_landmarks):", type(hand_landmarks))
        try:
            lm_list = list(hand_landmarks.landmark)
            print("len(hand_landmarks.landmark) =", len(lm_list))
            # imprime os 3 primeiros landmarks com x,y,z e tipos
            for i, lm in enumerate(lm_list[:5]):
                # tenta obter x,y,z com fallback
                x = getattr(lm, 'x', None)
                y = getattr(lm, 'y', None)
                z = getattr(lm, 'z', None)
                print(f"  lm[{i}]: x={x}, y={y}, z={z}, types: {type(x)}, {type(y)}, {type(z)}")
            # constrói pontos como o código faz
            pontos = []
            for lm in lm_list:
                x = getattr(lm, 'x', None)
                y = getattr(lm, 'y', None)
                z = getattr(lm, 'z', 0.0)  # se z faltar, considera 0.0
                if x is None or y is None:
                    print("  -> Encontrado landmark sem x/y — abortando este frame")
                    pontos = None
                    break
                pontos.extend([float(x), float(y), float(z)])
            if pontos is None:
                print("Frame inválido: landmarks sem coords x/y")
            else:
                print("Tamanho de 'pontos' calculado:", len(pontos))
                print("Primeiros 9 valores (3 pontos):", pontos[:9])
        except Exception as e:
            print("Erro ao inspecionar landmarks:", e)

        found = True
        break

    # show small feedback (opcional)
    if frame_idx % 200 == 0:
        print(f"lendo frame {frame_idx}... sem mão detectada ainda")

cap.release()
hands.close()

if not found:
    print("Nenhuma mão detectada em todo o vídeo.")

