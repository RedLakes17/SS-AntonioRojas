from transformers import pipeline



# Ejemplo de modelo preentrenado
generator = pipeline('text-generation', model='gpt2') #Llama a modelo preentrenado
generated_text = generator("Today is a beautiful day and", max_length=30) #Completa cadena de texto con un maximo de 30 caracteres
print(generated_text)




#Para crear datos artificiales
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader




# Se generan datos que siguen una curva sinosoidal
def generate_data(size=1000, sequence_length=10):
    data = np.sin(np.linspace(0, 10 * np.pi, size))  # Sine wave data
    sequences = [data[i:i+sequence_length] for i in range(size-sequence_length)]
    next_points = data[sequence_length:]
    return np.array(sequences), next_points




# Custom dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, next_points):
        self.sequences = sequences
        self.next_points = next_points

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.next_points[idx]





# Transformer Model (simplified for numerical data)
class TransformerModel(nn.Module):
    def __init__(self, input_size=1, sequence_length=10, num_layers=1, \
                 num_heads=2, dim_feedforward=512):
        super(TransformerModel, self).__init__()
        self.sequence_length = sequence_length
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size*sequence_length,
                           nhead=num_heads,
                           dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer,
                      num_layers=num_layers)
        self.fc_out = nn.Linear(input_size * sequence_length, 1)

    def forward(self, src):
        # Reshape to match the input dimensions
        src = src.reshape(-1, self.sequence_length, 1)  
        src = src.flatten(start_dim=1)
        src = src.unsqueeze(0)  # Add batch dimension
        out = self.transformer_encoder(src)
        out = out.squeeze(0)  # Remove batch dimension
        return self.fc_out(out)
    




# Prepare data
sequences, next_points = generate_data()
dataset = TimeSeriesDataset(sequences, next_points)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)





# Model
model = TransformerModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(9):  # Number of epochs
    for seq, next_point in dataloader:
        seq, next_point = seq.float(), next_point.float().unsqueeze(1)
        output = model(seq)
        loss = criterion(output, next_point)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")





# Predict the next point after a sequence
test_seq = torch.tensor(sequences[0]).float()
predicted_point = model(test_seq)
print("Predicted next point:", predicted_point.item())