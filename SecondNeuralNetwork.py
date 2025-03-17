#Red neuronal con 3 capas lineales
import torch
import torch.nn as nn


in_values=3
out_values=6
model=nn.Sequential(
    nn.Linear(in_values,5),
    nn.Linear(5,5),
    nn.Linear(5, out_values))
  

#More neurons=more parameters=higher capacity
#Una forma de saber la capacidad de un modelo es sabiendo el numero de parametros que contiene.
#Por ejemplo en la primera capa, el modelo tiene 3 pesos mas 1 bias por cada neurona de 4, i.e. 16 

total=0
for i in model.parameters():
    print(i.numel())
    total=total + i.numel()
print(total)

#Hay que balancear complejidad con eficienciaa