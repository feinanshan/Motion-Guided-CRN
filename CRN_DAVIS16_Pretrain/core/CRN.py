import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
import torchvision

import math



def Loss_calc(out, label):
    criterion = nn.BCELoss()
    return criterion(out,label)


class GCN(nn.Module):
    def __init__(self, inplanes, planes, ks=7):
        super(GCN, self).__init__()
        self.conv_l1 = nn.Conv2d(inplanes, planes, kernel_size=(ks, 1),
                                 padding=(ks/2, 0))

        self.conv_l2 = nn.Conv2d(planes, planes, kernel_size=(1, ks),
                                 padding=(0, ks/2))
        self.conv_r1 = nn.Conv2d(inplanes, planes, kernel_size=(1, ks),
                                 padding=(0, ks/2))
        self.conv_r2 = nn.Conv2d(planes, planes, kernel_size=(ks, 1),
                                 padding=(ks/2, 0))

        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)

        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)

        x = x_l + x_r

        x = self.relu(self.bn(x))

        return x


class RegionBottleneck(nn.Module):
    def __init__(self, planes):
        super(RegionBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)

    def forward(self, input):
        x = input[0]
        mask = input[1]
        residual = x
        x = self.conv1(x)
        x = self.relu1(self.bn1(x))
        x = self.conv2(x)
        x = self.relu2(self.bn2(x))
        x = x*mask
        out = residual + x
        return out


class Refine(nn.Module):
    def __init__(self, planes, step=1):
        super(Refine, self).__init__()
        self.rgConv_list = nn.ModuleList()
        for i in range(step):
            self.rgConv_list.append(RegionBottleneck(planes))

    def forward(self, input):
        x = input[0]
        mask = input[1]
        for i in range(len(self.rgConv_list)):
            x = self.rgConv_list[i]([x,mask])
        return x
        
        


class CRN(nn.Module):
    def __init__(self, mtype=101, num_classes=1):
        super(CRN, self).__init__()

        self.num_classes = num_classes
        self.num_medium = 32
        if  mtype == 50:
            print('####################################')
            print('Using CRN_Res50')
            print('####################################')
            resnet =torchvision.models.resnet50(pretrained=True)
        elif mtype == 101:
            print('####################################')
            print('Using CRN_Res101')
            print('####################################')
            resnet =torchvision.models.resnet101(pretrained=True)
        else:
            raise Exception('ResNet-50 or ResNet 101')

        self.conv1 = resnet.conv1
        self.bn0 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.gcn1 = GCN(2048, self.num_medium)
        self.gcn2 = GCN(1024, self.num_medium)
        self.gcn3 = GCN(512, self.num_medium)
        self.gcn4 = GCN(256, self.num_medium)
        self.gcn5 = GCN(64, self.num_medium)

        self.refine1 = Refine(self.num_medium,5)

        self.refine2_1 = Refine(self.num_medium,5)
        self.refine2_2 = Refine(self.num_medium,5)

        self.refine3_1 = Refine(self.num_medium,5)
        self.refine3_2 = Refine(self.num_medium,5)

        self.refine4_1 = Refine(self.num_medium,5)
        self.refine4_2 = Refine(self.num_medium,5)

        self.refine5_1 = Refine(self.num_medium,5)
        self.refine5_2 = Refine(self.num_medium,5)

        self.refine6 = Refine(self.num_medium,5)

        self.out1 = self._classifier(self.num_medium)
        self.out2 = self._classifier(self.num_medium)
        self.out3 = self._classifier(self.num_medium)
        self.out4 = self._classifier(self.num_medium)
        self.out5 = self._classifier(self.num_medium)
        self.out6 = self._classifier(self.num_medium)


    def _classifier(self, inplanes):
        return nn.Sequential(
            nn.Conv2d(inplanes, inplanes, 3, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, self.num_classes, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        iniMask = x[1]
        x = x[0] 
        input = x
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        conv_x = x
        x = self.maxpool(x)
        pool_x = x

        fm1 = self.layer1(x)   #1/4
        fm2 = self.layer2(fm1) #1/8
        fm3 = self.layer3(fm2) #1/16
        fm4 = self.layer4(fm3) #1/32

        fs1 = self.refine1([self.gcn1(fm4),iniMask])   #1/32
        mask1 = self.out1(fs1)

        mask2_0 = nn.Sigmoid()(F.interpolate(mask1, size=fm3.size()[2:],mode='bilinear', align_corners=True))
        gcfm2 = self.refine2_1([self.gcn2(fm3),mask2_0])   #1/16  
        fs2_0 = F.interpolate(fs1, size=fm3.size()[2:],mode='bilinear', align_corners=True) + gcfm2
        fs2 = self.refine2_2([fs2_0,mask2_0])
        mask2 = self.out2(fs2)   

        mask3_0 = nn.Sigmoid()(F.interpolate(mask2, size=fm2.size()[2:],mode='bilinear', align_corners=True))
        gcfm3 = self.refine3_1([self.gcn3(fm2),mask3_0])   #1/8       
        fs3_0 = F.interpolate(fs2, size=fm2.size()[2:],mode='bilinear', align_corners=True) + gcfm3 
        fs3 = self.refine3_2([fs3_0,mask3_0])
        mask3 = self.out3(fs3)

        mask4_0 = nn.Sigmoid()(F.interpolate(mask3, size=fm1.size()[2:],mode='bilinear', align_corners=True))      
        gcfm4 = self.refine4_1([self.gcn4(fm1),mask4_0])   #1/4
        fs4_0 = F.interpolate(fs3, size=fm1.size()[2:],mode='bilinear', align_corners=True) + gcfm4  
        fs4 = self.refine4_2([fs4_0,mask4_0]) 
        mask4 = self.out4(fs4)

        
        mask5_0 = nn.Sigmoid()(F.interpolate(mask4, size=conv_x.size()[2:],mode='bilinear', align_corners=True))   
        gcfm5 = self.refine5_1([self.gcn5(conv_x),mask5_0])   #1/2
        fs5_0 = F.interpolate(fs4, size=conv_x.size()[2:],mode='bilinear', align_corners=True) + gcfm5 
        fs5 = self.refine5_2([fs5_0,mask5_0]) 
        mask5 = self.out5(fs5)
         
        mask6_0 = nn.Sigmoid()(F.interpolate(mask5, size=input.size()[2:],mode='bilinear', align_corners=True))
        fs6_0 = F.interpolate(fs5, size=input.size()[2:],mode='bilinear', align_corners=True)
        fs6 = self.refine6([fs6_0,mask6_0]) 
        mask6 = self.out6(fs6)

        return mask6, mask5, mask4, mask3, mask2, mask1
