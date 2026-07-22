import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib


# Caminho da pasta onde estão os arquivos .npy
PASTA_DADOS = "dados_libras"


X = []  # Dados (vetores de 63 posições)
y = []  # Rótulos (letras)


# Carrega todos os arquivos .npy e extrai os dados
for arquivo in os.listdir(PASTA_DADOS):
   if arquivo.endswith(".npy"):
       caminho = os.path.join(PASTA_DADOS, arquivo)
       dados = np.load(caminho)
       X.append(dados)


       # O rótulo é a primeira letra do nome do arquivo: A_0.npy → "A"
       y.append(arquivo[0].upper())


# Converte para arrays
X = np.array(X)
y = np.array(y)


# Divide em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Cria e treina o modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)


# Avalia
acuracia = modelo.score(X_test, y_test)
print(f"Acurácia do modelo: {acuracia:.2%}")


# Salva o modelo
joblib.dump(modelo, "../modelo_libras.pkl")
print("Modelo salvo como modelo_libras.pkl")