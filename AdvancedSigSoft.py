#El proposito del Forward Pass es producri predicciones al pasar la informacion a traves de la neurona

import torch
import torch.nn as nn
import random


'''Binary clasification'''
Perros_caracteristicas=[] #Creamos registro de 5 perros con 6 caracteristicas
for i in range(5):
    Perro=[]
    for j in range(6):
        car=random.uniform(0,10)
        car=round(car, 2)
        Perro.append(car)
    Perros_caracteristicas.append(Perro)


#Creamos el modelo
Tensor_perros=torch.tensor(Perros_caracteristicas)
print(Tensor_perros)

Model=nn.Sequential(
    nn.Linear(6, 4),
    nn.Linear(4, 1),
    nn.Sigmoid() #Funcion de activacion
)

Output=Model(Tensor_perros)
print(Output) #Este output servira cuando actualicemos los bais y weights




'''Multiclass calssification'''
n_clases=3

ModelM=nn.Sequential(
    nn.Linear(6, 4),
    nn.Linear(4, n_clases),
    nn.Softmax(dim=-1)
)

Output=ModelM(Tensor_perros)
print(Output)




'''Regression. Predecir valores continuos'''
#Creamos el mismo modelo, pero sin funcion de activacion
Model3=nn.Sequential(
    nn.Linear(6,4),
    nn.Linear(4,1),
)
Output=Model3(Tensor_perros)
print(Output)





