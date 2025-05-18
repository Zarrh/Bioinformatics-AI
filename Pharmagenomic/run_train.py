import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

import json
from sklearn.metrics import confusion_matrix

from data_generation import generate_dataset
from encoding import encode_sample, idx2drug
from model import PharmaModel

# 1. Generazione dataset
raw_data = generate_dataset(n=10000)

# 2. Codifica dei dati in numeri
encoded_data = [encode_sample(sample) for sample in raw_data]
X, y = zip(*encoded_data)

# 3. Conversione in tensori
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# 4. Divisione in train/test
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# 5. Istanziazione del modello
model = PharmaModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 6. Training loop
epochs = 150
losses = []

for epoch in range(epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Salva il modello e la lista delle loss
torch.save(model.state_dict(), "trained_model.pth")

with open("loss_log.json", "w") as f:
    json.dump(losses, f)

# 7. Valutazione sul test set
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    predictions = torch.argmax(test_outputs, dim=1)
    accuracy = (predictions == y_test).float().mean()

# Salva predizioni
results = {
    "true": y_test.tolist(),
    "pred": predictions.tolist()
}

with open("prediction_vs_actual.json", "w") as f:
    json.dump(results, f)

print(f"\n🔍 Accuratezza sul test set: {accuracy.item() * 100:.2f}%")

# 8. Esempio predizione
def predict(sample):
    x_encoded, _ = encode_sample(sample)
    x_tensor = torch.tensor([x_encoded], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        output = model(x_tensor)
        pred_idx = torch.argmax(output, dim=1).item()
        return idx2drug[pred_idx]

if __name__ == "__main__":
    esempio = {
        "diagnosis": "Depressione",
        "CYP2D6": "IM",
        "CYP2C19": "UM",
    }
    pred = predict(esempio)
    print(f"\nPaziente esempio: {esempio}\n💊 Predizione farmaco per paziente esempio: {pred}")