import torch.nn as nn


class FCDiscriminator(nn.Module):

    def __init__(self, in_ch=2048, ndf=64):
        super(FCDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, ndf, kernel_size=(3, 3), stride=(1, 1),  padding=(1, 1))
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=(3, 3), stride=(1, 1),  padding=(1, 1))
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=(3, 3), stride=(1, 1),  padding=(1, 1))
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=(3, 3), stride=(1, 1),  padding=(1, 1))
        self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=(3, 3), stride=(1, 1),  padding=(1, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)

        return x
