import torch
import torch.nn as nn
import torch.nn.functional as F

class PharmaModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=16, output_dim=3):
        super(PharmaModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Test rapido del modello
if __name__ == "__main__":
    model = PharmaModel()
    
    # Esempio di input: [diagnosi=0, CYP2D6=1, CYP2C19=2]
    input_tensor = torch.tensor([[0.0, 1.0, 2.0]])  # batch di 1
    output = model(input_tensor)
    
    print("Output grezzo del modello:", output)
    print("Predizione (classe con prob. più alta):", torch.argmax(output, dim=1).item())
