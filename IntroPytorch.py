#Deep learning es una subrama del machine learning, ya que consta de multiples capas y estructuras anidadas muy complejas
import torch
#La unidad basica de Pytorch son los Tensor1es. Pueden ser creados a partir de una lista p un array numpy
Lista=[[1,2,5],[45,5,98],[23,1,76]]
Tensor1=torch.tensor(Lista)
print(Tensor1)
print(Tensor1.shape)
print(Tensor1.dtype)

#Podemos operar tensores si sus dimensiones son compatibles
Tensor2=torch.tensor([[3,34,5],[34,67,2],[234,5,87]])
print(Tensor1 + Tensor2)
print(Tensor1 * Tensor2)

#Multiplicacion de matrices
print(Tensor1 @ Tensor2)