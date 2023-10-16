import torch
from nets.BackBone.MobileNetv2 import mobilenetv2

if __name__ == '__main__':
    inputs = torch.randn((1, 3, 512, 512))
    
    model = mobilenetv2()
    low_feature, res= model(inputs)
    print(res.size())