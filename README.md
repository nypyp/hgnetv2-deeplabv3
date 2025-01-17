## Torch版的TransLab和DeepLabv3+全家桶
---

### 目录

1. [仓库更新 Top News](#仓库更新)
2. [相关仓库 Related code](#相关仓库)
3. [性能情况 Performance](#性能情况)
4. [所需环境 Environment](#所需环境)
5. [文件下载 Download](#文件下载)
6. [训练步骤 How2train](#训练步骤)
7. [预测步骤 How2predict](#预测步骤)
8. [评估步骤 miou](#评估步骤)
9. [参考资料 Reference](#Reference)

## Top News

**`2023-09`** : **新增TransLab分割头，可以通过设置pp参数切换**

TransLab是一款由仪酷智能科技有限公司开发的分割头，在这款分割头里面，我们将DeepLabv3基于传统卷积的空洞卷积 换成了基于Transformer的AIFI模块

~~玩Transformer玩的~~

**`2023-08`**:**在原作者基础上添加多个新款Backbone（HGNetv2,yolov8系列）**

如果在 新模型（HGNetv2 YOLOv8 MobileNetv3)有疑问或者建议 欢迎issue和PR

仪酷LabView工业AI推理插件工具包已经支持此项目包括最新主干在内的模型

如果需要原版代码 请访问https://github.com/bubbliiiing/deeplabv3-plus-pytorch

**`2022-04`**:**支持多GPU训练。**

**`2022-03`**:**进行大幅度更新、支持step、cos学习率下降法、支持adam、sgd优化器选择、支持学习率根据batch_size自适应调整。**  
BiliBili视频中的原仓库地址为：https://github.com/bubbliiiing/deeplabv3-plus-pytorch/tree/bilibili

**`2020-08`**:**创建仓库、支持多backbone、支持数据miou评估、标注数据处理、大量注释等。**

## 相关仓库

| 模型         | 路径                                                    |
|:-----------|:------------------------------------------------------|
| Unet       | https://github.com/bubbliiiing/unet-pytorch           |
| PSPnet     | https://github.com/bubbliiiing/pspnet-pytorch         |
| deeplabv3+ | https://github.com/bubbliiiing/deeplabv3-plus-pytorch |
| hrnet      | https://github.com/bubbliiiing/hrnet-pytorch          |

### 性能情况

|   训练数据集   |                                                             权值文件名称                                                              |   测试数据集   | 输入图片大小  | mIOU  | 
|:---------:|:-------------------------------------------------------------------------------------------------------------------------------:|:---------:|:-------:|:-----:| 
| VOC12+SBD | [deeplab_mobilenetv2.pth](https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/deeplab_mobilenetv2.pth) | VOC-Val12 | 512x512 | 72.59 | 
| VOC12+SBD |    [deeplab_xception.pth](https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/deeplab_xception.pth)    | VOC-Val12 | 512x512 | 76.95 | 
| VOC12+SBD |                  [deeplab_hgnetv2.pth](http://dl.aiblockly.com:8145/pretrained-model/seg/deeplab_hgnetv2.pth)                   | VOC-Val12 | 512x512 | 78.83 |
| VOC12+SBD |                  [translab_hgnetv2.pth](https://github.com/VIRobotics/hgnetv2-deeplabv3/releases/tag/v0.0.2-beta)                 | VOC-Val12 | 512x512 | 80.23 |

#### 目前该项目支持的主干网络有

MobileNetv2 MobileNetv3 XCeption HGNetv2(HGNet由百度开发，仪酷智能接入deeplab)，

YOLOv8(S和M尺寸，目前存在低mIOU的问题，不推荐)

#### 目前该项目支持的分割头有

官方Deeplabv3+的头（采用ASPP)

仪酷智能科技的TransLab头(采用AIFI Transformer) 

您可以自由的组合主干和分割头



### 所需环境

参看requirements.txt

### 文件下载

比较新的deeplab_HGNetv2由仪酷智能科技提供 [链接](http://dl.aiblockly.com:8145/pretrained-model/seg/deeplab_hgnetv2.pth)

```SHA256: D5DD6AB2556F87B8F03F12CCC14DCBEBADF01123003E1FBF3DB749D6477DBF8F```

训练所需的deeplab_mobilenetv2.pth和deeplab_xception.pth可在百度网盘中下载。     
链接: https://pan.baidu.com/s/1IQ3XYW-yRWQAy7jxCUHq8Q 提取码: qqq4

VOC拓展数据集的百度网盘如下：  
链接: https://pan.baidu.com/s/1vkk3lMheUm6IjTXznlg7Ng 提取码: 44mk

### 训练步骤

#### a、训练voc数据集

1、将我提供的voc数据集放入VOCdevkit中（无需运行voc_annotation.py）。  
2、在train.py中设置对应参数，默认参数已经对应voc数据集所需要的参数了，所以只要修改backbone,pp,和model_path即可。  
3、运行train.py进行训练。

#### b、训练自己的数据集

1、本文使用VOC格式进行训练。  
2、训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的SegmentationClass中。    
3、训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。    
4、在训练前利用voc_annotation.py文件生成对应的txt。    
5、在train.py文件夹下面，选择自己要使用的主干模型和下采样因子,支持的模型在预测步骤中有描述。下采样因子可以在8和16中选择。需要注意的是，预训练模型需要和主干模型相对应。   
6、注意修改train.py的num_classes为分类个数+1。    
7、运行train.py即可开始训练。

### 预测步骤

#### a、使用预训练权重

1、下载完库后解压，如果想用backbone为mobilenet的进行预测，直接运行predict.py就可以了；如果想要利用backbone为xception的进行预测，在百度网盘下载deeplab_xception.pth，放入model_data，修改deeplab.py的backbone和model_path之后再运行predict.py，输入。

```bash
img/street.jpg
```

可完成预测。    
2、在predict.py里面进行设置可以进行fps测试、整个文件夹的测试和video视频检测。

#### b、使用自己训练的权重

1、按照训练步骤训练。    
2、在deeplab.py文件里面，在如下部分修改model_path、num_classes、backbone使其对应训练好的文件；*
*model_path对应logs文件夹下面的权值文件，num_classes代表要预测的类的数量加1，backbone是所使用的主干特征提取网络**。

```python
_defaults = {
    # ----------------------------------------#
    #   model_path指向logs文件夹下的权值文件
    # ----------------------------------------#
    "model_path": 'model_data/deeplab_hgnetv2.pth',
    # ----------------------------------------#
    #   所需要区分的类的个数+1
    # ----------------------------------------#
    "num_classes": 21,
    # ----------------------------------------#
    #   所使用的的主干网络
    #   此处可选：mobilenet| xception | hgnetv2l | hgnetv2x | yolov8s | yolov8m | mobilenetv3s | mobilenetv3l
    # ----------------------------------------#
    "backbone": "hgnetv2l",
    # ----------------------------------------#
    #   输入图片的大小
    # ----------------------------------------#
    "input_shape": [512, 512],
    # ----------------------------------------#
    #   下采样的倍数，一般可选的为8和16
    #   与训练时设置的一样即可
    # ----------------------------------------#
    "downsample_factor": 16,
    # --------------------------------#
    #   blend参数用于控制是否
    #   让识别结果和原图混合
    # --------------------------------#
    "blend": True,
    # -------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    # -------------------------------#
    "cuda": True,
    # -------------------------------#
    #   使用何种头部 transformer代表使用TransLab 使用ASPP代表使用原版
    "pp":"transformer"
}
```

3、运行predict.py，输入

```bash
img/street.jpg
```

可完成预测。    
4、在predict.py里面进行设置可以进行fps测试、整个文件夹的测试和video视频检测。

### 评估步骤

1、设置get_miou.py里面的num_classes为预测的类的数量加1。  
2、设置get_miou.py里面的name_classes为需要去区分的类别。  
3、运行get_miou.py即可获得miou大小。

### Reference

https://github.com/ggyyzm/pytorch_segmentation  
https://github.com/bonlime/keras-deeplab-v3-plus  
https://github.com/bubbliiiing/deeplabv3-plus-pytorch  
https://github.com/ultralytics/ultralytics  
