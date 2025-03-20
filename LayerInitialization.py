#Los valores iniciales de las layers son muy imortantes

import torch
import torch.nn as nn

layer=nn.Linear(64,128)
nn.init.uniform_(layer.weight)

print(layer.weight.min(), layer.weight.max())

#Transfer learning se refiere a reutilizar un modelo entrenado para tareas similares. Para esto se guardan los pesos del modelo referencia y se cargan en el nuevo
layer2=nn.Linear(64,128)
torch.save(layer2, 'layer.pth')
new_layer=torch.load('layer.pth')

#Un tipo especial de transfer learning es llamado fine-tuning. Cargamos los peosos de un medelo previo y los usamos en el actual con un learning rate mas pequeño
#Podemos cargar solo una parte de los pesos y congelar algunas capas. La rule of thumb consiste en congelar las primeras capas y to fine-tune las capas mas proximas a la salida
model=nn.Sequential(
    nn.Linear(64,128),
    nn.Linear(128,256)
)
for name,param in model.named_parameters():
    if name == '0.weight':
        param.requires_grad=False