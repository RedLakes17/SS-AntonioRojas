#La prediccion es mala cuando la loss funtion es alta
#Para mejorar la prediccion debemos actualizar los pesos y bias. Esto lo hacemos con el back propagation
#Calculamos los gradientes de la funcion de perdida. Un gradiente respecto a cada capa del modelo

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

sample=torch.tensor([[2,65,14]], dtype=torch.float32)
target=torch.tensor([1], dtype=torch.long)

#Run a forwrd pass
Model=nn.Sequential(
    nn.Linear(3,8),
    nn.Linear(8,4),
    nn.Linear(4,2)
)
predition=Model(sample)

#Calculamos los gradientes de perdida
criterion=CrossEntropyLoss()
loss=criterion(predition, target)
loss.backward()



#Access each layer gradients
G1=Model[0].weight.grad
G2=Model[0].bias.grad
G3=Model[1].weight.grad
G4=Model[1].bias.grad

#Para actualizar los parametros manualmente (ejemplo con weight0)
lr=0.1 #learning rate
weight=Model[0].weight
weight_grad=G1

weight= weight - lr*weight_grad #le restamos al peso el producto del learning rate y el gradiente de si mismo




#Para funciones no convexas (con varios minimos) usaremos el gradient descent. Pytorch lo simplifica con optimizadores como es stocastic gradient descent SGD
import torch.optim as optim
optimizer=optim.SGD(Model.parameters(), lr=0.001)
#Perform parameter updates
optimizer.step()