import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.modules.block import NAMAttention,StripPooling,ACmix,GAMAttention,DySnakeConv 

from nets.modules import AIFI,Conv,TransformerLayer
class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.att = NAMAttention(dim_out*5)

    def forward(self, x):
        [b, c, row, col] = x.size()
        # -----------------------------------------#
        #   一共五个分支
        # -----------------------------------------#
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        # -----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        # -----------------------------------------#
        global_feature = StripPooling(x,(2,3),nn.BatchNorm2d,False)
        # global_feature = torch.mean(x, 2, True)
        # global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        # -----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        # -----------------------------------------#
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        attres = self.att(feature_cat)
        result = self.conv_cat(attres)
        return result




class TransEnc(nn.Module):
    def __init__(self, *args,dim_in=2048,dim_out=256,bn_mom=0.1,rate=1,**kwargs):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.__AIFI=AIFI(dim_in,1024,8)
        self.__tflayer=TransformerLayer(dim_in,8,256*3)
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)
        self.conv_cat = nn.Sequential(
            # TransformerLayer(dim_out*6, 8, dim_out),
            nn.Conv2d(dim_out*2+dim_in, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.att = NAMAttention(dim_out*2+dim_in,no_spatial=False,h_w=1024)
        #self.att = ACmix(736,736)
        #self.att = GAMAttention(688,688)
        #self.strpool = StripPooling(dim_in,(16,16),nn.BatchNorm2d,{'mode': 'bilinear', 'align_corners': True})

    def forward(self, x):
        [b, c, row, col] = x.size()
        x0=self.branch1(x)
        x1=self.__AIFI(x)
        #print(row,col)
        # x1=self.__tflayer(x1)
        #global_feature = self.strpool(x)
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        feature_cat = torch.cat([x0,x1, global_feature], dim=1)
        attres = self.att(feature_cat)
        result = self.conv_cat(attres)
        return result