import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

# ================================
# 1. Definir modelo (mesmo do treino)
# ================================
class GestureTransformer(nn.Module):
    def __init__(self, num_classes, input_dim=63, num_heads=4, num_layers=2, hidden_dim=128):
        super(GestureTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, N_FRAMES, 63)
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.mean(dim=0)
        return self.fc(x)

# ================================
# 2. Carregar modelo treinado
# ================================
N_FRAMES = 30
NUM_CLASSES = 26  # A-Z

model = GestureTransformer(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load("modelo_transformer.pth"))
model.eval()

# ================================
# 3. Função de normalização
# ================================
def normalizar_landmarks(pontos):
    landmarks = np.array(pontos).reshape(-1, 3)
    punho = landmarks[0]
    landmarks = landmarks - punho
    distancias = np.linalg.norm(landmarks, axis=1)
    max_dist = np.max(distancias)
    if max_dist > 0:
        landmarks = landmarks / max_dist
    return landmarks.flatten()

# ================================
# 4. Inicializar MediaPipe + câmera
# ================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
cap = cv2.VideoCapture(0)

classes = {i: chr(ord('A')+i) for i in range(26)}  # 0->A, 1->B, ...

gravando = False
sequencia = []
predicao = None
conf = 0.0

print("Use ESPAÇO para começar/parar a gravação do gesto")
print("Pressione ESC para sair")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = hands.process(rgb)

    # Captura landmarks se estiver gravando
    if resultados.multi_hand_landmarks and gravando:
        hand_landmarks = resultados.multi_hand_landmarks[0]
        pontos = []
        for lm in hand_landmarks.landmark:
            pontos.extend([lm.x, lm.y, lm.z])
        pontos_norm = normalizar_landmarks(pontos)
        sequencia.append(pontos_norm)

    # Mostrar status na tela
    if gravando:
        cv2.putText(frame, "Gravando...", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    elif predicao is not None:
        letra, conf = predicao
        cv2.putText(frame, f"Letra: {letra} ({conf:.1f}%)", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("Reconhecimento de Libras", frame)

    key = cv2.waitKey(1) & 0xFF

    # Tecla ESC para sair
    if key == 27:
        break

    # Tecla espaço: alterna entre gravando e parando
    if key == 32:  # espaço
        if not gravando:
            # Começa nova gravação
            sequencia = []
            gravando = True
        else:
            # Para gravação e faz a predição
            gravando = False
            if len(sequencia) >= N_FRAMES:
                # Pega só os últimos N_FRAMES
                sequencia = sequencia[-N_FRAMES:]
            else:
                # Faz padding se gravou menos que N_FRAMES
                padding = np.zeros((N_FRAMES - len(sequencia), 63))
                sequencia = np.vstack([sequencia, padding])

            entrada = torch.tensor([sequencia], dtype=torch.float32)
            with torch.no_grad():
                saida = model(entrada)
                probs = torch.softmax(saida, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                letra = classes[pred]
                conf = probs[0, pred].item() * 100

            predicao = (letra, conf)

cap.release()
cv2.destroyAllWindows()
