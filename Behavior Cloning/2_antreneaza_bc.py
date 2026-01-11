import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import TensorDataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128 
EPOCHS = 100 

print(f"--> [Hardware] Antrenăm folosind: {DEVICE}")

class BehaviorCloningNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BehaviorCloningNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

def antreneaza_iteratia(dataset_path, nume_iteratie):
    print(f"\n>>> Încep antrenarea pentru: {nume_iteratie} ({dataset_path})")
    
    if not os.path.exists(dataset_path):
        print(f"EROARE: Nu găsesc {dataset_path}.")
        return []

    with open(dataset_path, "rb") as f:
        data = pickle.load(f)
    
    observations = torch.FloatTensor(data["observations"]).to(DEVICE)
    actions = torch.FloatTensor(data["actions"]).to(DEVICE)

    if len(observations.shape) > 2:
        observations = observations.reshape(observations.shape[0], -1)

    input_dim = observations.shape[1]
    output_dim = actions.shape[1] 

    dataset = TensorDataset(observations, actions)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = BehaviorCloningNet(input_dim, output_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss() 

    loss_history = []

    for epoch in range(EPOCHS):
        epoch_loss = 0
        for batch_obs, batch_act in loader:
            optimizer.zero_grad()               
            predictii = model(batch_obs)        
            loss = criterion(predictii, batch_act) 
            loss.backward()                    
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"    [Epoca {epoch+1}/{EPOCHS}] Loss: {avg_loss:.5f}")

    nume_fisier_model = f"model_{nume_iteratie.lower().replace(' ', '_')}.pth"
    torch.save(model.state_dict(), nume_fisier_model)
    print(f"    -> Model salvat ca: {nume_fisier_model}")
    
    return loss_history

if __name__ == "__main__":
    loss_1 = antreneaza_iteratia("dataset_slab.pkl", "Iteratia 1")
    loss_2 = antreneaza_iteratia("dataset_mediu.pkl", "Iteratia 2")
    loss_3 = antreneaza_iteratia("dataset_expert.pkl", "Iteratia 3")

    if loss_1 and loss_3:
        plt.figure(figsize=(10, 6))
        plt.plot(loss_1, label='Iteratia 1 (Random)', linestyle='--', color='red')
        plt.plot(loss_2, label='Iteratia 2 (Mediu)', color='orange')
        plt.plot(loss_3, label='Iteratia 3 (Expert)', color='green', linewidth=2)
        
        plt.title("Performanța Antrenării (Scăderea Erorii)")
        plt.xlabel("Epoci")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig("rezultat_final_grafic.png")
        print("\n=== Grafic Loss Generat ===")