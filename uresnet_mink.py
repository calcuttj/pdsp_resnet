import torch
import torch.nn as nn
from resnet_mink import SEBlock
import MinkowskiEngine as ME

from MinkowskiEngine import (MinkowskiConvolution as MEConv,
                             MinkowskiConvolutionTranspose as MEConvTr,
                             MinkowskiBatchNorm as MEBatchNorm,
                             MinkowskiReLU as MEReLU,
                             MinkowskiMaxPooling as MEMaxPool,
                             MinkowskiAvgPooling as MEAvgPool,
                             MinkowskiGlobalAvgPooling as MEGlobalAvgPool,
                             MinkowskiBroadcastAddition as MEAdd,
                             MinkowskiBroadcastAddition as MEMult,
                             MinkowskiLinear as MELinear,
                             MinkowskiSigmoid as MESigmoid,
                            )

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

    self.no_residual = (f_out < f_in)

    self.first_batch = (MEBatchNorm(f_in) if f_in > 1 else None)
    self.first_relu = MEReLU()

    self.layers = nn.Sequential(
      self.first_batch,
      self.first_relu,
      MEConv(f_in, f_out, kernel_size=3, stride=1, dimension=2),
      MEBatchNorm(f_out),
      MEReLU(),
      MEConv(f_out, f_out, kernel_size=3, stride=1, dimension=2),
      SEBlock(f_out),
    )


  def forward(self, x):

    if self.no_residual:
      y = self.layers(x)
    else:
      residual_portion = self.first_relu(self.first_batch(x))
      if self.residual is not None:
        residual_portion = self.residual(residual_portion)
      y = self.layers(x) + residual_portion

    return y

   

class Model(ME.MinkowskiNetwork):
  def __init__(self, nfinal=3):
    super().__init__(D=2)

    #initial
    #self.stage1 = nn.Sequential(
    #  MEConv(1, 64, kernel_size=7, stride=2, dimension=2),
    #  MEMaxPool(kernel_size=3, stride=2, dimension=2),
    #)

    #contraction
    self.step1 = nn.Sequential(
      #MEConv(1, 64, kernel_size=7, stride=2, dimension=2),
      MEConv(1, 64, kernel_size=3, stride=2, dimension=2),
      #MEMaxPool(kernel_size=3, stride=2, dimension=2),
      SubBlock(64, 64),
    )

    self.step2 = nn.Sequential(
      MEMaxPool(kernel_size=2, stride=2, dimension=2),
      SubBlock(64, 128),
    )

    self.step3 = nn.Sequential(
      MEMaxPool(kernel_size=2, stride=2, dimension=2),
      SubBlock(128, 256),
    )

    self.step4 = nn.Sequential(
      MEMaxPool(kernel_size=2, stride=2, dimension=2),
      SubBlock(256, 512),
    )
    
    self.step5 = nn.Sequential(
      MEMaxPool(kernel_size=2, stride=2, dimension=2),
      SubBlock(512, 1024),
    )

    #expansion
    self.upconv1 = MEConvTr(1024, 512, kernel_size=2, stride=2, dimension=2)
    self.step6 = SubBlock(1024, 512)

    self.upconv2 = MEConvTr(512, 256, kernel_size=2, stride=2, dimension=2)
    self.step7 = SubBlock(512, 256)

    self.upconv3 = MEConvTr(256, 128, kernel_size=2, stride=2, dimension=2)
    self.step8 = SubBlock(256, 128)

    self.upconv4 = MEConvTr(128, 64, kernel_size=2, stride=2, dimension=2)
    self.step9 = SubBlock(128, 64)

    self.upconv5 = MEConvTr(64, 64, kernel_size=3, stride=2, dimension=2)

    #final
    self.final = nn.Sequential(
      MEConv(64, nfinal, kernel_size=1, stride=1, dimension=2),
      ME.MinkowskiSoftmax(),
    )

  def forward(self, x):
    #contracting
    out1 = self.step1(x) 
    out2 = self.step2(out1)
    out3 = self.step3(out2)
    out4 = self.step4(out3)
    out = self.step5(out4)

    #expanding
    out = self.upconv1(out, coordinates=out4.coordinate_map_key)
    out = ME.cat(out, out4)
    #out += out4
    out = self.step6(out)

    out = self.upconv2(out, coordinates=out3.coordinate_map_key)
    out = ME.cat(out, out3)
    #out += out3
    out = self.step7(out)

    out = self.upconv3(out, coordinates=out2.coordinate_map_key)
    out = ME.cat(out, out2)
    #out += out2
    out = self.step8(out)

    out = self.upconv4(out, coordinates=out1.coordinate_map_key)
    out = ME.cat(out, out1)
    #out += out1
    out = self.step9(out)
    out = self.upconv5(out)
    out = self.final(out)
    return out
