import cv2
import mediapipe as mp
import numpy as np
import os
from normalizar import normalizar_landmarks

# inicializar mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# pasta pra salvar os dados
PASTA_SAIDA = "dados_libras"
if not os.path.exists(PASTA_SAIDA):
    os.makedirs(PASTA_SAIDA)

# qual letra vai ser coletada e qual pasta (ex: C:\Users\USER\PycharmProjects\PythonProject\videos\A)
letra = input("Digite a letra (A-Z) que deseja coletar: ").upper()
pasta_videos = input("Digite o caminho da pasta com os vídeos: ")

contador = 0


# percorrer todos os arquivos de vídeo da pasta
for arquivo in os.listdir(pasta_videos):
    if arquivo.endswith((".mp4", ".avi", ".mov", ".mkv")):  # só vídeos
        caminho_video = os.path.join(pasta_videos, arquivo)
        cap = cv2.VideoCapture(caminho_video)

        print(f"Processando vídeo: {arquivo}")

        sequencia = [] #lista com os frames do vídeo (63 de x,y,z)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # terminou o vídeo

            frame = cv2.flip(frame, 1)  # espelhar imagem
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultados = hands.process(rgb)

            if resultados.multi_hand_landmarks:
                for hand_landmarks in resultados.multi_hand_landmarks:
                    pontos = []
                    for lm in hand_landmarks.landmark:
                        pontos.extend([lm.x, lm.y, lm.z])

                        #normalizar os pontos
                        #pontos_normalizados = normalizar_landmarks(pontos)
                        #sequencia.append(pontos_normalizados)  # adiciona o frame na sequencia

                if len(pontos) == 63:
                    # normalizar os pontos
                   pontos_normalizados = normalizar_landmarks(pontos)
                   sequencia.append(pontos_normalizados)
                else:
                  print(f"⚠️ Frame ignorado: {len(pontos)} valores encontrados")
            else:
              print("⚠️ Nenhuma mão detectada neste frame.")



            # Mostra o vídeo em tempo real
            cv2.imshow("Processando vídeos - Libras", frame)

            # ESC para parar
            if cv2.waitKey(1) & 0xFF == 29:
                break

        cap.release()
        if len(sequencia) > 0: #pra não salvar arquivo vazio
            sequencia = np.array(sequencia)  # sequencia.shape = (n_frames, 63)
            nome_arquivo = f"{PASTA_SAIDA}/{letra}_{contador}.npy"
            np.save(nome_arquivo, sequencia)
            print(f"Sequência salva: {nome_arquivo} com {sequencia.shape[0]} frames")
            contador += 1

cv2.destroyAllWindows()
print(f"Coleta concluída! {contador} amostras salvas para a letra {letra}.")