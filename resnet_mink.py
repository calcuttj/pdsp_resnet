import torch
import torch.nn as nn
import MinkowskiEngine as ME
from MinkowskiEngine import (MinkowskiConvolution as MEConv,
                             MinkowskiBatchNorm as MEBatchNorm,
                             MinkowskiDropout as MEDropout,
                             MinkowskiReLU as MEReLU,
                             MinkowskiMaxPooling as MEMaxPool,
                             MinkowskiAvgPooling as MEAvgPool,
                             MinkowskiGlobalAvgPooling as MEGlobalAvgPool,
                             MinkowskiBroadcastAddition as MEAdd,
                             MinkowskiBroadcastAddition as MEMult,
                             MinkowskiLinear as MELinear,
                             MinkowskiSigmoid as MESigmoid,
                            )


class SEBlock(nn.Module):
  def __init__(self, f_in, r=16, dropout=False):
    super().__init__()
    self.layers = nn.Sequential(
      MEGlobalAvgPool(),
      MELinear(f_in, f_in // r),
      MEReLU(),
      (MEDropout(p=.2) if dropout else nn.Identity()),
      MELinear(f_in // r, f_in),
      MESigmoid(),
      (MEDropout(p=.2) if dropout else nn.Identity()),
    )

    self.scale = MEMult()

  def forward(self, x):
    return self.scale(x, self.layers(x)) 

class SubBlock(nn.Module):
  def __init__(self, f_in, f_out, dropout=False):
    super().__init__()
    self.f_in=f_in
    self.f_out=f_out
    self.dropout=dropout
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
      (MEDropout(p=.2) if self.dropout else nn.Identity()),
      self.first_relu,
      MEConv(f_in, f_out, kernel_size=3, stride=initial_stride, dimension=2),
      MEBatchNorm(f_out),
      (MEDropout(p=.2) if self.dropout else nn.Identity()),
      MEReLU(),
      MEConv(f_out, f_out, kernel_size=3, stride=1, dimension=2),
      SEBlock(f_out, dropout=True),
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
      dropout=False,
  ):
    super().__init__(D=2)

    self.stage1 = nn.Sequential(
      MEConv(1, 64, kernel_size=7, stride=2, dimension=2),
      MEMaxPool(kernel_size=3, stride=2, dimension=2),
    )

    self.stage2 = nn.Sequential(
      SubBlock(64, 64, dropout=dropout),
      SubBlock(64, 64, dropout=dropout),
      SubBlock(64, 64, dropout=dropout),
    )

    self.stage3 = nn.Sequential(
      SubBlock(64, 128, dropout=dropout),
      SubBlock(128, 128, dropout=dropout),
      SubBlock(128, 128, dropout=dropout),
      SubBlock(128, 128, dropout=dropout),
    )

    self.stage4 = nn.Sequential(
      SubBlock(128, 256, dropout=dropout),
      SubBlock(256, 256, dropout=dropout),
      SubBlock(256, 256, dropout=dropout),
      SubBlock(256, 256, dropout=dropout),
      SubBlock(256, 256, dropout=dropout),
      SubBlock(256, 256, dropout=dropout),
    )

    self.stage5 = nn.Sequential(
      SubBlock(256, 512, dropout=dropout),
      SubBlock(512, 512, dropout=dropout),
      SubBlock(512, 512, dropout=dropout),
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
