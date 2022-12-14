import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.checkpoint import checkpoint
import torch.nn.init as init
import numpy as np
from torchsummary import summary

'''
    Ordinary UNet Conv Block
'''
class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.leaky_relu):
        super(UNetConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_size, out_size, kernel_size, stride=1, padding=1)
        self.gn = nn.GroupNorm(num_groups=8,num_channels=out_size)
        self.conv2 = nn.Conv3d(out_size, out_size, kernel_size, stride=1, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=8,num_channels=out_size)
        self.activation = activation


        init.xavier_uniform_(self.conv.weight, gain = np.sqrt(2.0))
        init.constant_(self.conv.bias,0)
        init.xavier_uniform_(self.conv2.weight, gain = np.sqrt(2.0))
        init.constant_(self.conv2.bias,0)
    def forward(self, x):
        out = self.activation(self.gn(self.conv(x)))
        out = self.activation(self.gn2(self.conv2(out)))

        return out


'''    
 two-layer residual unit: two conv with GN/relu and identity mapping
'''
class residualUnit(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3,stride=1, padding=1, activation=F.relu, SE=False):
        super(residualUnit, self).__init__()
        self.conv1 = nn.Conv3d(in_size, out_size, kernel_size, stride=1, padding=1)
        init.xavier_uniform_(self.conv1.weight, gain = np.sqrt(2.0)) #or gain=1
        init.constant_(self.conv1.bias, 0)
        self.conv2 = nn.Conv3d(out_size, out_size, kernel_size, stride=1, padding=1)
        init.xavier_uniform_(self.conv2.weight, gain = np.sqrt(2.0)) #or gain=1
        init.constant_(self.conv2.bias, 0)
        self.activation = activation
        self.gn1 = nn.GroupNorm(num_groups=8,num_channels=out_size)
        self.gn2 = nn.GroupNorm(num_groups=8,num_channels=out_size)
        self.in_size = in_size
        self.out_size = out_size
        self.SE = SE
        # if SE:
        #     self.seblock = ChannelSELayer3D(num_channels=out_size)
        if in_size != out_size:
            self.convX = nn.Conv3d(in_size, out_size, kernel_size=1, stride=1, padding=0)
            self.gnX = nn.GroupNorm(num_groups=8,num_channels=out_size)

    def forward(self, x):
        out1 = self.activation(self.gn1(self.conv1(x)))
        out2 = self.activation(self.gn2(self.conv2(out1)))
        if self.SE:
            out = self.seblock(out2)
        else:
            out = out2
        if self.in_size!=self.out_size:
            bridge = self.activation(self.gnX(self.convX(x)))
        elif self.in_size==self.out_size:
            bridge = x
        output = torch.add(out, bridge)

        return output


'''
    Ordinary UNet-Up Conv Block
'''
class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu, space_dropout=False):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose3d(in_size, out_size, 2, stride=2)
        self.bnup = nn.BatchNorm3d(out_size)
        self.conv = nn.Conv3d(in_size, out_size, kernel_size, stride=1, padding=1)
        self.bn = nn.BatchNorm3d(out_size)
        self.conv2 = nn.Conv3d(out_size, out_size, kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_size)
        self.activation = activation
        init.xavier_uniform(self.up.weight, gain = np.sqrt(2.0))
        init.constant(self.up.bias,0)
        init.xavier_uniform(self.conv.weight, gain = np.sqrt(2.0))
        init.constant(self.conv.bias,0)
        init.xavier_uniform(self.conv2.weight, gain = np.sqrt(2.0))
        init.constant(self.conv2.bias,0)

    def center_crop(self, layer, target_size):
        batch_size, n_channels, layer_width, layer_height, layer_depth = layer.size()
        xy1 = (layer_width - target_size) // 2
        return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size)]

    def forward(self, x, bridge):
        up = self.up(x)
        up = self.activation(self.bnup(up))
        crop1 = self.center_crop(bridge, up.size()[2])
        out = torch.cat([up, crop1], 1)

        out = self.activation(self.bn(self.conv(out)))
        out = self.activation(self.bn2(self.conv2(out)))

        return out



'''
    Ordinary Residual UNet-Up Conv Block
'''
class UNetUpResBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu, space_dropout=False,SE=False):
        super(UNetUpResBlock, self).__init__()
        self.up = nn.ConvTranspose3d(in_size, out_size, 2, stride=2)
        self.gnup = nn.GroupNorm(num_groups=8,num_channels=out_size)

        init.xavier_uniform_(self.up.weight, gain = np.sqrt(2.0))
        init.constant_(self.up.bias,0)

        self.activation = activation

        self.resUnit = residualUnit(in_size, out_size, kernel_size = kernel_size,SE=SE)

    def center_crop(self, layer, target_size):
        batch_size, n_channels, layer_width, layer_height, layer_depth = layer.size()
        xy1 = (layer_width - target_size) // 2
        return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size), xy1:(xy1 + target_size)]

    def forward(self, x, bridge):
        #print 'x.shape: ',x.shape
        up = self.activation(self.gnup(self.up(x)))
        #crop1 = self.center_crop(bridge, up.size()[2])
        #print 'up.shape: ',up.shape, ' crop1.shape: ',crop1.shape
        crop1 = bridge
        out = torch.cat([up, crop1], 1)

        out = self.resUnit(out)
        # out = self.activation(self.bn2(self.conv2(out)))

        return out


'''
    Ordinary UNet
'''
class UNet(nn.Module):
    def __init__(self, in_channel = 4, n_classes = 4):
        super(UNet, self).__init__()
#         self.imsize = imsize

        self.activation = F.relu

        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)
        self.pool3 = nn.MaxPool3d(2)
        # self.pool4 = nn.MaxPool3d(2)


        self.conv_block1_64 = UNetConvBlock(in_channel, 32)
        self.conv_block64_128 = UNetConvBlock(32, 64)
        self.conv_block128_256 = UNetConvBlock(64, 128)
        self.conv_block256_512 = UNetConvBlock(128, 256)
        # self.conv_block512_1024 = UNetConvBlock(512, 1024)
        # this kind of symmetric design is awesome, it automatically solves the number of channels during upsamping
        # self.up_block1024_512 = UNetUpBlock(1024, 512)
        self.up_block512_256 = UNetUpBlock(256, 128)
        self.up_block256_128 = UNetUpBlock(128, 64)
        self.up_block128_64 = UNetUpBlock(64, 32)

        self.last = nn.Conv3d(32, n_classes, 1, stride=1)


    def forward(self, x):
#         print 'line 70 ',x.size()
        block1 = self.conv_block1_64(x)
        pool1 = self.pool1(block1)

        block2 = self.conv_block64_128(pool1)
        pool2 = self.pool2(block2)

        block3 = self.conv_block128_256(pool2)
        pool3 = self.pool3(block3)

        block4 = self.conv_block256_512(pool3)
        # pool4 = self.pool4(block4)
        #
        # block5 = self.conv_block512_1024(pool4)
        #
        # up1 = self.up_block1024_512(block5, block4)

        up2 = self.up_block512_256(block4, block3)

        up3 = self.up_block256_128(up2, block2)

        up4 = self.up_block128_64(up3, block1)

        return nn.Softmax(dim=1)(self.last(up4))


'''
    Ordinary ResUNet
'''

class ResUNet(nn.Module):
    def __init__(self, in_channel=4, n_classes=4, filter = 2,init_weights=True,CheckPoint=False, SE=False):
        super(ResUNet, self).__init__()
        #         self.imsize = imsize


        filters1 = [16,32,64,128]
        filters2 = [32, 64, 128, 256]
        filters3 = [16, 32, 64, 128, 256]

        self.checkpoint = CheckPoint
        self.filter = filter

        if filter==1:
            filters = filters1
        elif filter==2:
            filters = filters2
        elif filter == 3:
            filters = filters3

        self.activation = F.relu

        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)
        self.pool3 = nn.MaxPool3d(2)
        # self.pool4 = nn.MaxPool3d(2)

        self.conv_block1_64 = UNetConvBlock(in_channel, filters[0])
        self.conv_block64_128 = residualUnit(filters[0], filters[1],SE=SE)
        self.conv_block128_256 = residualUnit(filters[1], filters[2],SE=SE)
        self.conv_block256_512 = residualUnit(filters[2], filters[3],SE=SE)
        if filter==3:
            self.conv_block512_1024 = residualUnit(filters[3], filters[4],SE=SE)
            # this kind of symmetric design is awesome, it automatically solves the number of channels during upsamping
            self.up_block1024_512 = UNetUpResBlock(filters[4], filters[3],SE=SE)
        self.up_block512_256 = UNetUpResBlock(filters[3], filters[2],SE=SE)
        self.up_block256_128 = UNetUpResBlock(filters[2], filters[1],SE=SE)
        self.up_block128_64 = UNetUpResBlock(filters[1], filters[0],SE=SE)

        self.last = nn.Conv3d(filters[0], n_classes, 1, stride=1)
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        #         print 'line 70 ',x.size()
        if self.checkpoint:
            inputs = x + torch.zeros(1, dtype=x.dtype, device=x.device, requires_grad=True)
            block1 = checkpoint(self.conv_block1_64,inputs)
            pool1 = self.pool1(block1)

            block2 = checkpoint(self.conv_block64_128,pool1)
            pool2 = self.pool2(block2)

            block3 = checkpoint(self.conv_block128_256,pool2)
            pool3 = self.pool3(block3)

            block4 = checkpoint(self.conv_block256_512,pool3)
            if self.filter==3:
                pool4 = self.pool4(block4)

                block5 = checkpoint(self.conv_block512_1024,pool4)

                up1 = checkpoint(self.up_block1024_512,block5, block4)

                up2 = checkpoint(self.up_block512_256, up1, block3)
            else:
                up2 = checkpoint(self.up_block512_256,block4, block3)

            up3 = checkpoint(self.up_block256_128,up2, block2)

            up4 = checkpoint(self.up_block128_64,up3, block1)

            out = checkpoint(self.last,up4)
            out = nn.Sigmoid()(out)
        else:
            block1 = self.conv_block1_64(x)
            pool1 = self.pool1(block1)

            block2 = self.conv_block64_128(pool1)
            pool2 = self.pool2(block2)

            block3 = self.conv_block128_256(pool2)
            pool3 = self.pool3(block3)

            block4 = self.conv_block256_512(pool3)
            # pool4 = self.pool4(block4)
            #
            # block5 = self.conv_block512_1024(pool4)
            #
            # up1 = self.up_block1024_512(block5, block4)

            up2 = self.up_block512_256(block4, block3)

            up3 = self.up_block256_128(up2, block2)

            up4 = self.up_block128_64(up3, block1)

            out = self.last(up4)
            out = nn.Sigmoid()(out)



        return out

'''
    ResUNet en--conv
'''
class ResUNet_conv(nn.Module):
    def __init__(self, in_channel=4, n_classes=4,init_weights=True,CheckPoint=False, SE=False):
        super(ResUNet_conv, self).__init__()
        #         self.imsize = imsize

        filters2 = [32, 64, 128, 256]
        self.checkpoint = CheckPoint
        filters = filters2


        self.activation = F.relu

        self.down1 = nn.Conv3d(filters[0], filters[1], kernel_size=3,padding=1, stride=2)
        self.down2 = nn.Conv3d(filters[1], filters[2], kernel_size=3,padding=1, stride=2)
        self.down3 = nn.Conv3d(filters[2], filters[3], kernel_size=3,padding=1, stride=2)

        self.conv_block1_64 = UNetConvBlock(in_channel, filters[0])
        self.conv_block64_128 = residualUnit(filters[1], filters[1],SE=SE)
        self.conv_block128_256 = residualUnit(filters[2], filters[2],SE=SE)
        self.conv_block256_512 = residualUnit(filters[3], filters[3],SE=SE)

        self.up_block512_256 = UNetUpResBlock(filters[3], filters[2],SE=SE)
        self.up_block256_128 = UNetUpResBlock(filters[2], filters[1],SE=SE)
        self.up_block128_64 = UNetUpResBlock(filters[1], filters[0],SE=SE)

        self.last = nn.Conv3d(filters[0], n_classes, 1, stride=1)
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        #         print 'line 70 ',x.size()
        if self.checkpoint:
            inputs = x + torch.zeros(1, dtype=x.dtype, device=x.device, requires_grad=True)
            block1 = checkpoint(self.conv_block1_64,inputs)
            pool1 = self.down1(block1)

            block2 = checkpoint(self.conv_block64_128,pool1)
            pool2 = self.down2(block2)

            block3 = checkpoint(self.conv_block128_256,pool2)
            pool3 = self.down3(block3)

            block4 = checkpoint(self.conv_block256_512,pool3)
            up2 = checkpoint(self.up_block512_256, block4, block3)

            up3 = checkpoint(self.up_block256_128,up2, block2)

            up4 = checkpoint(self.up_block128_64,up3, block1)

            out = checkpoint(self.last,up4)
            out = nn.Sigmoid()(out)
        else:
            block1 = self.conv_block1_64(x)
            pool1 = self.down1(block1)

            block2 = self.conv_block64_128(pool1)
            pool2 = self.down2(block2)

            block3 = self.conv_block128_256(pool2)
            pool3 = self.down3(block3)

            block4 = self.conv_block256_512(pool3)
            # pool4 = self.pool4(block4)
            #
            # block5 = self.conv_block512_1024(pool4)
            #
            # up1 = self.up_block1024_512(block5, block4)

            up2 = self.up_block512_256(block4, block3)

            up3 = self.up_block256_128(up2, block2)

            up4 = self.up_block128_64(up3, block1)

            out = self.last(up4)
            out = nn.Sigmoid()(out)

        return out

'''
    ResUNet en--conv
'''
class ResUNet_4bottom(nn.Module):
    def __init__(self, in_channel=4, n_classes=4,init_weights=True,CheckPoint=False, SE=False):
        super(ResUNet_4bottom, self).__init__()
        #         self.imsize = imsize

        filters = [32, 64, 128, 256]
        # filters = [16, 32, 64, 128]
        self.checkpoint = CheckPoint

        self.activation = F.relu

        self.down1 = nn.Conv3d(filters[0], filters[1], kernel_size=3,padding=1, stride=2)
        self.down2 = nn.Conv3d(filters[1], filters[2], kernel_size=3,padding=1, stride=2)
        self.down3 = nn.Conv3d(filters[2], filters[3], kernel_size=3,padding=1, stride=2)

        self.conv_block1_32 = UNetConvBlock(in_channel, filters[0])
        self.conv_block64_1 = residualUnit(filters[1], filters[1],SE=SE)
        self.conv_block64_2 = residualUnit(filters[1], filters[1], SE=SE)
        self.conv_block128_1 = residualUnit(filters[2], filters[2],SE=SE)
        self.conv_block128_2 = residualUnit(filters[2], filters[2], SE=SE)
        self.conv_block256_1 = residualUnit(filters[3], filters[3],SE=SE)
        self.conv_block256_2 = residualUnit(filters[3], filters[3], SE=SE)
        self.conv_block256_3 = residualUnit(filters[3], filters[3], SE=SE)
        self.conv_block256_4 = residualUnit(filters[3], filters[3], SE=SE)

        self.up_block256_128 = UNetUpResBlock(filters[3], filters[2],SE=SE)
        self.up_block128_64 = UNetUpResBlock(filters[2], filters[1],SE=SE)
        self.up_block64_32 = UNetUpResBlock(filters[1], filters[0],SE=SE)

        self.last = nn.Conv3d(filters[0], n_classes, 1, stride=1)
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        #         print 'line 70 ',x.size()
        if self.checkpoint:
            inputs = x + torch.zeros(1, dtype=x.dtype, device=x.device, requires_grad=True)
            block1 = checkpoint(self.conv_block1_32,inputs)
            pool1 = self.down1(block1)

            block2 = checkpoint(self.conv_block64_1,pool1)
            block2 = checkpoint(self.conv_block64_2, block2)
            pool2 = self.down2(block2)

            block3 = checkpoint(self.conv_block128_1,pool2)
            block3 = checkpoint(self.conv_block128_2, block3)
            pool3 = self.down3(block3)

            block4 = checkpoint(self.conv_block256_1,pool3)
            block4 = checkpoint(self.conv_block256_2, block4)
            block4 = checkpoint(self.conv_block256_3, block4)
            block4 = checkpoint(self.conv_block256_4, block4)

            up2 = checkpoint(self.up_block256_128, block4, block3)

            up3 = checkpoint(self.up_block128_64,up2, block2)

            up4 = checkpoint(self.up_block64_32,up3, block1)

            out = checkpoint(self.last,up4)
            out = nn.Sigmoid()(out)
        else:
            block1 = self.conv_block1_32(x)
            pool1 = self.down1(block1)

            block2 = self.conv_block64_1(pool1)
            block2 = self.conv_block64_2(block2)
            pool2 = self.down2(block2)

            block3 = self.conv_block128_1(pool2)
            block3 = self.conv_block128_2(block3)
            pool3 = self.down3(block3)

            block4 = self.conv_block256_1(pool3)
            block4 = self.conv_block256_2(block4)
            block4 = self.conv_block256_3(block4)
            block4 = self.conv_block256_4(block4)

            up2 = self.up_block256_128(block4, block3)

            up3 = self.up_block128_64(up2, block2)

            up4 = self.up_block64_32(up3, block1)

            out = self.last(up4)
            out = nn.Softmax(dim=1)(out)

        return out
class ResUNet_4bottom2(nn.Module):
    def __init__(self, in_channel=4, n_classes=3,init_weights=True,CheckPoint=False, SE=False):
        super(ResUNet_4bottom2, self).__init__()
        #         self.imsize = imsize

        filters = [32, 64, 128, 256]
        # filters = [16, 32, 64, 128]
        self.checkpoint = CheckPoint

        self.activation = F.relu

        self.down1 = nn.Conv3d(filters[0], filters[1], kernel_size=3,padding=1, stride=2)
        self.down2 = nn.Conv3d(filters[1], filters[2], kernel_size=3,padding=1, stride=2)
        self.down3 = nn.Conv3d(filters[2], filters[3], kernel_size=3,padding=1, stride=2)

        self.conv_block1_32 = UNetConvBlock(in_channel, filters[0])
        self.conv_block64_1 = residualUnit(filters[1], filters[1],SE=SE)
        self.conv_block64_2 = residualUnit(filters[1], filters[1], SE=SE)
        self.conv_block128_1 = residualUnit(filters[2], filters[2],SE=SE)
        self.conv_block128_2 = residualUnit(filters[2], filters[2], SE=SE)
        self.conv_block256_1 = residualUnit(filters[3], filters[3],SE=SE)
        self.conv_block256_2 = residualUnit(filters[3], filters[3], SE=SE)
        self.conv_block256_3 = residualUnit(filters[3], filters[3], SE=SE)
        self.conv_block256_4 = residualUnit(filters[3], filters[3], SE=SE)

        self.up_block256_128 = UNetUpResBlock(filters[3], filters[2],SE=SE)
        self.up_block128_64 = UNetUpResBlock(filters[2], filters[1],SE=SE)
        self.up_block64_32 = UNetUpResBlock(filters[1], filters[0],SE=SE)

        self.last = nn.Conv3d(filters[0], n_classes, 1, stride=1)
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        #         print 'line 70 ',x.size()
        if self.checkpoint:
            inputs = x + torch.zeros(1, dtype=x.dtype, device=x.device, requires_grad=True)
            block1 = checkpoint(self.conv_block1_32,inputs)
            pool1 = self.down1(block1)

            block2 = checkpoint(self.conv_block64_1,pool1)
            block2 = checkpoint(self.conv_block64_2, block2)
            pool2 = self.down2(block2)

            block3 = checkpoint(self.conv_block128_1,pool2)
            block3 = checkpoint(self.conv_block128_2, block3)
            pool3 = self.down3(block3)

            block4 = checkpoint(self.conv_block256_1,pool3)
            block4 = checkpoint(self.conv_block256_2, block4)
            block4 = checkpoint(self.conv_block256_3, block4)
            block4 = checkpoint(self.conv_block256_4, block4)

            up2 = checkpoint(self.up_block256_128, block4, block3)

            up3 = checkpoint(self.up_block128_64,up2, block2)

            up4 = checkpoint(self.up_block64_32,up3, block1)

            out = checkpoint(self.last,up4)
            out = nn.Sigmoid()(out)
        else:
            block1 = self.conv_block1_32(x)
            pool1 = self.down1(block1)

            block2 = self.conv_block64_1(pool1)
            block2 = self.conv_block64_2(block2)
            pool2 = self.down2(block2)

            block3 = self.conv_block128_1(pool2)
            block3 = self.conv_block128_2(block3)
            pool3 = self.down3(block3)

            block4 = self.conv_block256_1(pool3)
            block4 = self.conv_block256_2(block4)
            block4 = self.conv_block256_3(block4)
            block4 = self.conv_block256_4(block4)

            up2 = self.up_block256_128(block4, block3)

            up3 = self.up_block128_64(up2, block2)

            up4 = self.up_block64_32(up3, block1)

            out = self.last(up4)
            out = nn.Sigmoid()(out)

        return out


class SegClass(nn.Module):
    def __init__(self,n_classes):
        super(SegClass, self).__init__()
        #         self.imsize = imsize

        self.stage1 = ResUNet_conv(n_classes=n_classes,filter=1, CheckPoint=True, SE=True)
        self.stage2 = ResUNet_conv(in_channel=4+n_classes,n_classes=n_classes,filter=2, CheckPoint=True, SE=True)

    def forward(self, x):
        out1 = self.stage1(x)
        out = self.stage2(torch.cat((x,out1), dim=1))

        return out1, out


'''
    UNet (lateral connection) with long-skip residual connection (from 1st to last layer)
'''
class UNet_LRes(nn.Module):
    def __init__(self, in_channel = 1, n_classes = 4):
        super(UNet_LRes, self).__init__()
#         self.imsize = imsize

        self.activation = F.relu

        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)
        self.pool3 = nn.MaxPool3d(2)
        # self.pool4 = nn.MaxPool3d(2)

        self.conv_block1_64 = UNetConvBlock(in_channel, 32)
        self.conv_block64_128 = UNetConvBlock(32, 64)
        self.conv_block128_256 = UNetConvBlock(64, 128)
        self.conv_block256_512 = UNetConvBlock(128, 256)
        # self.conv_block512_1024 = UNetConvBlock(512, 1024)
        # this kind of symmetric design is awesome, it automatically solves the number of channels during upsamping
        # self.up_block1024_512 = UNetUpBlock(1024, 512)
        self.up_block512_256 = UNetUpBlock(256, 128)
        self.up_block256_128 = UNetUpBlock(128, 64)
        self.up_block128_64 = UNetUpBlock(64, 32)

        self.last = nn.Conv3d(32, n_classes, 1, stride=1)


    def forward(self, x, res_x):
#         print 'line 70 ',x.size()
        block1 = self.conv_block1_64(x)
        pool1 = self.pool1(block1)

        block2 = self.conv_block64_128(pool1)
        pool2 = self.pool2(block2)

        block3 = self.conv_block128_256(pool2)
        pool3 = self.pool3(block3)

        block4 = self.conv_block256_512(pool3)
        # pool4 = self.pool4(block4)

        # block5 = self.conv_block512_1024(pool4)
        #
        # up1 = self.up_block1024_512(block5, block4)

        up2 = self.up_block512_256(block4, block3)

        up3 = self.up_block256_128(up2, block2)

        up4 = self.up_block128_64(up3, block1)

        last = self.last(up4)
        #print 'res_x.shape is ',res_x.shape,' and last.shape is ',last.shape
        if len(res_x.shape) == 3:
            res_x = res_x.unsqueeze(1)
        out = torch.add(last, res_x)

        #print 'out.shape is ',out.shape
        return out


'''
    ResUNet (lateral connection) with long-skip residual connection (from 1st to last layer)
'''


class ResUNet_LRes(nn.Module):
    def __init__(self, in_channel=1, n_classes=4, dp_prob=0):
        super(ResUNet_LRes, self).__init__()
        #         self.imsize = imsize

        self.activation = F.relu

        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)
        self.pool3 = nn.MaxPool3d(2)
        # self.pool4 = nn.MaxPool3d(2)

        self.conv_block1_64 = UNetConvBlock(in_channel, 32)
        self.conv_block64_128 = residualUnit(32, 64)
        self.conv_block128_256 = residualUnit(64, 128)
        self.conv_block256_512 = residualUnit(128, 256)
        # self.conv_block512_1024 = residualUnit(512, 1024)
        # this kind of symmetric design is awesome, it automatically solves the number of channels during upsamping
        # self.up_block1024_512 = UNetUpResBlock(1024, 512)
        self.up_block512_256 = UNetUpResBlock(256, 128)
        self.up_block256_128 = UNetUpResBlock(128, 64)
        self.up_block128_64 = UNetUpResBlock(64, 32)
        self.Dropout = nn.Dropout3d(p=dp_prob)
        self.last = nn.Conv3d(32, n_classes, 1, stride=1)

    def forward(self, x, res_x):
        #         print 'line 70 ',x.size()
        block1 = self.conv_block1_64(x)
        # print 'block1.shape: ', block1.shape
        pool1 = self.pool1(block1)
        # print 'pool1.shape: ', block1.shape
        pool1_dp = self.Dropout(pool1)
        # print 'pool1_dp.shape: ', pool1_dp.shape
        block2 = self.conv_block64_128(pool1_dp)
        pool2 = self.pool2(block2)

        pool2_dp = self.Dropout(pool2)

        block3 = self.conv_block128_256(pool2_dp)
        pool3 = self.pool3(block3)

        pool3_dp = self.Dropout(pool3)

        block4 = self.conv_block256_512(pool3_dp)
        # pool4 = self.pool4(block4)
        #
        # pool4_dp = self.Dropout(pool4)
        #
        # # block5 = self.conv_block512_1024(pool4_dp)
        #
        # up1 = self.up_block1024_512(block5, block4)

        up2 = self.up_block512_256(block4, block3)

        up3 = self.up_block256_128(up2, block2)

        up4 = self.up_block128_64(up3, block1)

        last = self.last(up4)
        # print 'res_x.shape is ',res_x.shape,' and last.shape is ',last.shape
        if len(res_x.shape) == 3:
            res_x = res_x.unsqueeze(1)
        out = torch.add(last, res_x)

        # print 'out.shape is ',out.shape
        return out



'''
    Discriminator for the reconstruction project
'''
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        #you can make abbreviations for conv and fc, this is not necessary
        #class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = nn.Conv3d(1,32,9)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32,64,5)
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64,64,5)
        self.bn3 = nn.BatchNorm3d(64)
        self.fc1 = nn.Linear(64*4*4,512)
        #self.bn3= nn.BatchNorm1d(6)
        self.fc2 = nn.Linear(512,64)
        self.fc3 = nn.Linear(64,1)


    def forward(self,x):
#         print 'line 114: x shape: ',x.size()
        #x = F.max_pool3d(F.relu(self.bn1(self.conv1(x))),(2,2,2))#conv->relu->pool
        x = F.max_pool3d(F.relu(self.conv1(x)),(2,2,2))#conv->relu->pool

        x = F.max_pool3d(F.relu(self.conv2(x)),(2,2,2))#conv->relu->pool

        x = F.max_pool3d(F.relu(self.conv3(x)),(2,2,2))#conv->relu->pool

        #reshape them into Vector, review ruturned tensor shares the same data but have different shape, same as reshape in matlab
        x = x.view(-1,self.num_of_flat_features(x))
        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        #x = F.sigmoid(x)
        #print 'min,max,mean of x in 0st layer',x.min(),x.max(),x.mean()
        return x

    def num_of_flat_features(self,x):
        size=x.size()[1:]#we donot consider the batch dimension
        num_features=1
        for s in size:
            num_features*=s
        return num_features

# test
def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # data = torch.ones((6,4,128,128,128)).cuda()
    model = ResUNet_conv(CheckPoint=True)
    model.cuda()
    # out = dis(data)
    summary(model, (4, 128, 128, 128), batch_size=1, device='cuda')
    # print(out.shape)

if __name__ == "__main__":
    main()