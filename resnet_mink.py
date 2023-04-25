import torch
import torch.nn as nn
import MinkowskiEngine as ME
from MinkowskiEngine import (MinkowskiConvolution as MEConv,
                             MinkowskiBatchNorm as MEBatchNorm,
                             MinkowskiReLU as MEReLU,
                             MinkowskiMaxPooling as MEMaxPool,
                             MinkowskiAvgPooling as MEAvgPool,
                             MinkowskiAvgPooling as MEAvgPool,
                             MinkowskiGlobalAvgPooling as MEGlobalAvgPool,
                             MinkowskiBroadcastAddition as MEAdd,
                             MinkowskiBroadcastAddition as MEMult,
                             MinkowskiLinear as MELinear,
                             MinkowskiSigmoid as MESigmoid,
                            )


class SEBlock(nn.Module):
  def __init__(self, f_in, r=16):
    super().__init__()
    self.layers = nn.Sequential(
      MEGlobalAvgPool(),
      MELinear(f_in, f_in // r),
      MEReLU(),
      MELinear(f_in // r, f_in),
      MESigmoid(),
    )

    self.scale = MEMult()

  def forward(self, x):
    return self.scale(x, self.layers(x)) 

class SubBlock(nn.Module):
  def __init__(self, f_in, f_out):
    super().__init__()
    self.f_in=f_in
    self.f_out=f_out
    #self.k=k ##Need?

    self.residual = (
      MEConv(f_in, f_out, kernel_size=1, stride=2, dimension=2) if f_out > f_in
      else None
    )

    initial_stride = 2 if f_out > f_in else 1

    self.first_batch = MEBatchNorm(f_in)
    self.first_relu = MEReLU()

    self.layers = nn.Sequential(
      self.first_batch,
      self.first_relu,
      MEConv(f_in, f_out, kernel_size=3, stride=initial_stride, dimension=2),
      MEBatchNorm(f_out),
      MEReLU(),
      MEConv(f_out, f_out, kernel_size=3, stride=1, dimension=2),
      SEBlock(f_out),
    )


  def forward(self, x):
    residual_portion = self.first_relu(self.first_batch(x))
    if self.residual is not None:
      residual_portion = self.residual(residual_portion)

    y = self.layers(x) + residual_portion

    return y

   

class Model(ME.MinkowskiNetwork):
  def __init__(
      self,
  ):
    super().__init__(D=2)

    self.stage1 = nn.Sequential(
      MEConv(1, 64, kernel_size=7, stride=2, dimension=2),
      MEMaxPool(kernel_size=3, stride=2, dimension=2),
    )

    self.stage2 = nn.Sequential(
      SubBlock(64, 64),
      SubBlock(64, 64),
      SubBlock(64, 64),
    )

    self.stage3 = nn.Sequential(
      SubBlock(64, 128),
      SubBlock(128, 128),
      SubBlock(128, 128),
      SubBlock(128, 128),
    )

    self.stage4 = nn.Sequential(
      SubBlock(128, 256),
      SubBlock(256, 256),
      SubBlock(256, 256),
      SubBlock(256, 256),
      SubBlock(256, 256),
      SubBlock(256, 256),
    )

    self.stage5 = nn.Sequential(
      SubBlock(256, 512),
      SubBlock(512, 512),
      SubBlock(512, 512),
    )

    self.global_avg_pool = ME.MinkowskiGlobalMaxPooling() 
    self.linear = MELinear(512, 4)
    self.activation = ME.MinkowskiSoftmax()

  def forward(self, x):
    y = self.stage1(x)
    y = self.stage2(y)
    y = self.stage3(y)
    y = self.stage4(y)
    y = self.stage5(y)
    y = self.global_avg_pool(y)
    y = self.linear(y)
    y = self.activation(y)
    return y
