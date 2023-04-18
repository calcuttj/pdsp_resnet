import torch
import torch.nn as nn
import sparseconvnet as scn

class Model(nn.Module):
  def __init__(self, resnet_filters=[16, 32, 64, 96]):
    nn.Module.__init__(self)

    layers = [['b', i, 2, 2] for i in resnet_filters]
    layers[0][-1] -= 1

    self.final_size = 2*resnet_filters[-1]
      
    self.sparseModel = scn.Sequential(
      #2x2, 1 input feature, 64 outputs, kernel size 7, no bias
      scn.SubmanifoldConvolution(2, 1, resnet_filters[0], 7, False),
      #2x2, pool size 3, stride 2
      scn.MaxPooling(2, 3, 2),
      #2x2 dimension, 16 inputs (from above conv),


      scn.SparseResNet(2, resnet_filters[0], layers),
      #scn.SparseResNet(2, 64, [#start layers
      #      ['b', 64, 2, 1],
      #      ['b', 128, 2, 2],
      #      #['b', 48, 2, 2],
      #      #['b', 96, 2, 2]]),
      #      ['b', 256, 2, 2],
      #      #['b', 512, 2, 2]]),
      #]),
      scn.Convolution(
          2, resnet_filters[-1], self.final_size, 3, 1, False),

      scn.BatchNormReLU(self.final_size),
      scn.SparseToDense(2, self.final_size)
    )
    self.spatial_size= self.sparseModel.input_spatial_size(torch.LongTensor([1, 1]))
    self.inputLayer = scn.InputLayer(2,self.spatial_size,2)
    self.linear = nn.Linear(self.final_size, 4)
    self.activate = nn.Softmax(dim=1)

  def forward(self, x):
      x = self.inputLayer(x)
      x = self.sparseModel(x)
      x = x.view(-1, self.final_size)
      x = self.linear(x)
      x = self.activate(x)
      return x
