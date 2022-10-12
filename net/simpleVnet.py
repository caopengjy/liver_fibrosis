import torch
import torch.nn as nn
import torch.nn.functional as F


class simpleVnet(nn.Module):
    """

    共9498260个可训练的参数, 接近九百五十万
    """
    def __init__(self, training):
        super().__init__()

        self.training = training

        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(1, 16, 3, 1, padding=1),
            nn.PReLU(16),

            nn.Conv3d(16, 16, 3, 1, padding=1),
            nn.PReLU(16),
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),
            # 这个就是上卷积了
            nn.Conv3d(64, 64, 3, 1, padding=2, dilation=2),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=4, dilation=4),
            nn.PReLU(64),
        )

        self.encoder_stage4 = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, padding=3, dilation=3),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=4, dilation=4),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=5, dilation=5),
            nn.PReLU(128),
        )

        self.decoder_stage1 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),
        )

        self.decoder_stage2 = nn.Sequential(
            nn.Conv3d(128 + 64, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),
        )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv3d(64 + 32, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv3d(32 + 16, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv3d(16, 32, 2, 2),
            nn.PReLU(32)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 2, 2),
            nn.PReLU(64)
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv3d(64, 128, 3, 1, padding=1),
            nn.PReLU(128)
        )

        # self.down_conv4 = nn.Sequential(
        #     nn.Conv3d(128, 256, 3, 1, padding=1),
        #     nn.PReLU(256)
        # )
        #
        # self.up_conv2 = nn.Sequential(
        #     nn.ConvTranspose3d(256, 128, 2, 2),
        #     nn.PReLU(128)
        # )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, 2),
            nn.PReLU(64)
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2),
            nn.PReLU(32)
        )

        # 最后大尺度下的映射（256*256），下面的尺度依次递减
        self.map4 = nn.Sequential(
            nn.Conv3d(32, 1, 1, 1),
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear'),
            nn.Sigmoid()
        )

        # 128*128 尺度下的映射
        self.map3 = nn.Sequential(
            nn.Conv3d(64, 1, 1, 1),
            nn.Upsample(scale_factor=(2, 4, 4), mode='trilinear'),
            nn.Sigmoid()
        )

        # 64*64 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Conv3d(128, 1, 1, 1),
            # 下面参数是放大倍数与差值方法
            nn.Upsample(scale_factor=(4, 8, 8), mode='trilinear'),
            nn.Sigmoid()
        )

        # 32*32 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Conv3d(256, 1, 1, 1),
            nn.Upsample(scale_factor=(8, 16, 16), mode='trilinear'),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        # 这个确实是对位sum element-wise sum
        # 其实好像就是残差网络？
        # 16个channel应该每一个都回加上院士channel
        long_range1 = self.encoder_stage1(inputs) + inputs  # 16*16*256*256

        short_range1 = self.down_conv1(long_range1)  # 32*8*128*128

        long_range2 = self.encoder_stage2(short_range1) + short_range1  # 32*8*128
        long_range2 = F.dropout(long_range2, 0.4, self.training)

        short_range2 = self.down_conv2(long_range2)  # 64*4*64

        long_range3 = self.encoder_stage3(short_range2) + short_range2  # 64*4*64
        long_range3 = F.dropout(long_range3, 0.4, self.training)

        short_range3 = self.down_conv3(long_range3)  # 128*4*64

        # long_range4 = self.encoder_stage4(short_range3) + short_range3 # 128*4*64
        # long_range4 = F.dropout(long_range4, 0.3, self.training)
        #
        # short_range4 = self.down_conv4(long_range4) # 256*2*32

        # outputs = self.decoder_stage1(long_range3) + short_range3
        # outputs = F.dropout(outputs, 0.3, self.training)
        #
        # # output1 = self.map1(outputs)
        #
        # short_range6 = self.up_conv2(outputs)

        outputs = self.decoder_stage2(torch.cat([short_range3, long_range3], dim=1)) + short_range3
        outputs = F.dropout(outputs, 0.4, self.training)

        # output2 = self.map2(outputs)

        short_range7 = self.up_conv3(outputs)

        outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1)) + short_range7
        outputs = F.dropout(outputs, 0.4, self.training)

        # output3 = self.map3(outputs)

        short_range8 = self.up_conv4(outputs)

        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1)) + short_range8

        output4 = self.map4(outputs)

        # if self.training is True:
        #     return output1, output2, output3, output4
        # else:
        return output4


def init(module):
    if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
        nn.init.kaiming_normal(module.weight.data, 0.25)
        nn.init.constant(module.bias.data, 0)


net = simpleVnet(training=True)
net.apply(init)

net = net.cuda()
data = torch.randn((1, 1, 16, 256, 256)).cuda()
print(net)

# 输出数据维度检查
res=net(data)
for item in res:
    print(item.shape)
    #print(item.size())

# 计算网络参数
print('net total parameters:', sum(param.numel() for param in net.parameters()))
