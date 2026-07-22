import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ==========================
# 1. Carregar dataset
# ==========================
X = np.load("X_transformer.npy")   # (n_amostras, N_FRAMES, 63)
y = np.load("y_transformer.npy")   # (n_amostras,)

# Converter para tensores do PyTorch
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Criar Dataset e DataLoader
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ==========================
# 2. Definir modelo Transformer
# ==========================
class GestureTransformer(nn.Module):
    def __init__(self, num_classes, input_dim=63, num_heads=4, num_layers=2, hidden_dim=128):
        super(GestureTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, N_FRAMES, 63)
        x = self.embedding(x)  # (batch, N_FRAMES, hidden_dim)
        x = x.permute(1, 0, 2)  # (N_FRAMES, batch, hidden_dim)
        x = self.transformer(x)  # (N_FRAMES, batch, hidden_dim)
        x = x.mean(dim=0)  # média no tempo -> (batch, hidden_dim)
        return self.fc(x)  # (batch, num_classes)

# ==========================
# 3. Instanciar modelo, loss e otimizador
# ==========================
num_classes = len(set(y.numpy()))  # número de classes do dataset
model = GestureTransformer(num_classes=num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ==========================
# 4. Loop de treino
# ==========================
EPOCHS = 10
for epoch in range(EPOCHS):
    total_loss = 0
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(dataloader):.4f}")

# ==========================
# 5. Salvar modelo treinado
# ==========================
torch.save(model.state_dict(), "modelo_transformer.pth")
print("Modelo salvo em modelo_transformer.pth ✅")
