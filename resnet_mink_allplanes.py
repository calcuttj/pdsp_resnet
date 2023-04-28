from resnet_mink import SubBlock
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


class Stage1(nn.Module):
  def __init__(self):
    super().__init__()

    self.stage = nn.Sequential(
      MEConv(1, 64, kernel_size=7, stride=2, dimension=2),
      MEMaxPool(kernel_size=3, stride=2, dimension=2),
    )

  def forward(self, x):
    return self.stage(x)

class StageN(nn.Module):
  def __init__(self, f_in, f_out, depth, dropout=False):
    super().__init__()

    #TODO -- Assert depth > 1?

    self.stage = nn.Sequential(
      SubBlock(f_in, f_out, dropout=dropout),
      *[SubBlock(f_out, f_out, dropout=dropout) for i in range(depth-1)]
    )

  def forward(self, x):
    return self.stage(x)

class Model(ME.MinkowskiNetwork):
  def __init__(
      self,
      dropout=False,
  ):
    super().__init__(D=2)

    self.stage1_plane0 = Stage1() 
    self.stage2_plane0 = StageN(64, 64, 3, dropout=dropout)
    self.stage3_plane0 = StageN(64, 128, 4, dropout=dropout)
    self.stage4_plane0 = StageN(128, 256, 6, dropout=dropout)
    self.stage5_plane0 = StageN(256, 512, 3, dropout=dropout)
    self.global_avg_pool_plane0 = ME.MinkowskiGlobalMaxPooling() 
    self.linear_plane0 = MELinear(512, 128)
    self.relu_plane0 = MEReLU()

    self.stage1_plane1 = Stage1() 
    self.stage2_plane1 = StageN(64, 64, 3, dropout=dropout)
    self.stage3_plane1 = StageN(64, 128, 4, dropout=dropout)
    self.stage4_plane1 = StageN(128, 256, 6, dropout=dropout)
    self.stage5_plane1 = StageN(256, 512, 3, dropout=dropout)
    self.global_avg_pool_plane1 = ME.MinkowskiGlobalMaxPooling() 
    self.linear_plane1 = MELinear(512, 128)
    self.relu_plane1 = MEReLU()

    self.stage1_plane2 = Stage1() 
    self.stage2_plane2 = StageN(64, 64, 3, dropout=dropout)
    self.stage3_plane2 = StageN(64, 128, 4, dropout=dropout)
    self.stage4_plane2 = StageN(128, 256, 6, dropout=dropout)
    self.stage5_plane2 = StageN(256, 512, 3, dropout=dropout)
    self.global_avg_pool_plane2 = ME.MinkowskiGlobalMaxPooling() 
    self.linear_plane2 = MELinear(512, 128)
    self.relu_plane2 = MEReLU()


    self.linear_mix0 = nn.Linear(384, 128)
    self.linear_mix1 = nn.Linear(128, 4)
    self.activation = nn.Softmax(dim=1)

  def forward(self, x0, x1, x2):
    y0 = self.stage1_plane0(x0)
    y0 = self.stage2_plane0(y0)
    y0 = self.stage3_plane0(y0)
    y0 = self.stage4_plane0(y0)
    y0 = self.stage5_plane0(y0)
    y0 = self.global_avg_pool_plane0(y0)
    y0 = self.linear_plane0(y0)
    y0 = self.relu_plane0(y0)

    y1 = self.stage1_plane1(x1)
    y1 = self.stage2_plane1(y1)
    y1 = self.stage3_plane1(y1)
    y1 = self.stage4_plane1(y1)
    y1 = self.stage5_plane1(y1)
    y1 = self.global_avg_pool_plane1(y1)
    y1 = self.linear_plane1(y1)
    y1 = self.relu_plane1(y1)

    y2 = self.stage1_plane2(x2)
    y2 = self.stage2_plane2(y2)
    y2 = self.stage3_plane2(y2)
    y2 = self.stage4_plane2(y2)
    y2 = self.stage5_plane2(y2)
    y2 = self.global_avg_pool_plane2(y2)
    y2 = self.linear_plane2(y2)
    y2 = self.relu_plane2(y2)

    y = torch.cat((y0.features, y1.features, y2.features), dim=1)
    y = self.linear_mix0(y)
    y = self.linear_mix1(y)
    y = self.activation(y)
    return y
