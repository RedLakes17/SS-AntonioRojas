#Sigmoid: Binary classification
#Softmax: Multi-class calssification

#Permiten aprender otros tipos de relaciones ademas de las lineales

import torch
import torch.nn as nn

#Sigmoid. Devuelve un valor entre 0 y 1
input=torch.tensor([[6]])
sigmoid=nn.Sigmoid()
output=sigmoid(input)
print(output)


#Softmax.
input_T=torch.tensor([[4.3,6.1,2.3]])
softmax_probabilidades=nn.Softmax(dim=-1) #Se aplica en la ultima dimension
output=softmax_probabilidades(input_T)
print(output)

