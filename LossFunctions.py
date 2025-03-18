#Las funciones de perdida nos dicen que tan bueno es nuestro modelo durante el entrenamiento
#Lo que queremos es minimizar la perdida
#La funcion de perdida toma como entradas F(y,y^) donde y es the ground truth y y^ es la prediccion
#Usamos One-hot encoding para transformar the ground truth a un tensor. Por ejemplo si y=0 -> y_T=[1,0,0] debido a que hay tres clases (funcion softmax)

'''One-hot encoding'''
import torch
import torch.nn.functional as F
Ten0=torch.tensor(0)#Clase esperada (ground truth)
Ten1=torch.tensor(1)
Ten2=torch.tensor(2)

print(F.one_hot(Ten0, num_classes=3))
print(F.one_hot(Ten1, num_classes=3))
print(F.one_hot(Ten2, num_classes=3))


'''Pasemos nuestra ground truth y prediccion a una loss function'''
#Escogemos 0
scores=torch.tensor([-5.2,4.6,0.8]) #y^ Supuesta prediccion por una red neuronal cualquiera
GroundTruth=torch.tensor([1,0,0]) #y

from torch.nn import CrossEntropyLoss
LossFunction=CrossEntropyLoss()
#Pasamos y, y^ a la loss function
print('The loss value is:',LossFunction(scores.double(), GroundTruth.double()))