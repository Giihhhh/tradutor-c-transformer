import numpy as np

def normalizar_landmarks(pontos):

    # array format : [x1, y1, z1, x2, y2, z2, ..., x21, y21, z21]
    # new format- matriz: 21 pontos por 3 coord.
    landmarks = np.array(pontos).reshape(-1, 3)


    #diminuir pelo punho (normalizar x e y)
    punho = landmarks[0]
    landmarks = landmarks - punho

    #dividir pra escalar (normalizar z)
    # distância euclidiana fórmula 3D (np.linalg.norm)

    distancias = np.linalg.norm(landmarks, axis=1)
    max_dist = np.max(distancias) #distancia max como ref. para o tamanho da mão

    if max_dist > 0:
        landmarks = landmarks / max_dist

    return landmarks.flatten()  # volta para (63,)