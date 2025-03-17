#Creemos una red sin hidden layers. Linear model, fully connected model
import torch
import torch.nn as nn

#Creamos un tensor o capa de entrada
InLayer=torch.tensor([3,6,3.4])

#Creamos la linear layer(flechitas)
LinearLayer=nn.Linear(in_features=3, out_features=2)

#Capa de salida
OutLayer=LinearLayer(InLayer)
print(OutLayer)

#Parametros de la capa lineal
print('Los pesos para cada neurona',LinearLayer.weight)
print('Los vias para cada neurona',LinearLayer.bias)


