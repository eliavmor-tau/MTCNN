from torch import nn


class PNet(nn.Module):
    
    def __init__(self):
        super(PNet, self).__init__()
        # Define P-Net architecture
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.prelu = nn.PReLU()

        self.conv2 = nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3, stride=1)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)

        # Bounding box regression layer
        self.conv4_0 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1, stride=1)
        # Classification layer
        self.conv4_1 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=1, stride=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.prelu(self.pool1(self.conv1(x)))
        x = self.prelu(self.conv2(x))
        x = self.prelu(self.conv3(x))
        bbox_layer = self.prelu(self.conv4_0(x)).reshape((batch_size, -1))
        classification_layer = self.conv4_1(x).reshape((batch_size, -1))
        return {"bbox_pred": bbox_layer, "y_pred": classification_layer}


class RNet(nn.Module):

    def __init__(self):
        super(RNet, self).__init__()
        # Define P-Net architecture
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=28, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.prelu = nn.PReLU()
        self.relue = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=28, out_channels=48, kernel_size=3, stride=1)

        self.conv3 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=2, stride=1)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)

        # Bounding box regression layer
        self.linear0_0 = nn.Linear(in_features=128, out_features=4)
        # Classification layer
        self.linear0_1 = nn.Linear(in_features=128, out_features=2)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.prelu(self.pool1(self.conv1(x)))
        x = self.prelu(self.pool1(self.conv2(x)))
        x = self.prelu(self.conv3(x))
        x = self.prelu(self.conv4(x))
        x = x.view((batch_size, 128))
        bbox_layer = self.relue(self.linear0_0(x)).view((batch_size, -1))
        classification_layer = self.linear0_1(x).view((batch_size, -1))
        return {"bbox_pred": bbox_layer, "y_pred": classification_layer}


class ONet(nn.Module):

    def __init__(self):
        super(ONet, self).__init__()
        # Define P-Net architecture
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.prelu = nn.PReLU()
        self.relue = nn.ReLU()

        # 23x23x32
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)

        # 10x10x64
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)

        # 4x4x64
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=1)

        # Bounding box regression layer
        self.linear0_0 = nn.Linear(in_features=256, out_features=4)
        # Classification layer
        self.linear0_1 = nn.Linear(in_features=256, out_features=2)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.prelu(self.pool1(self.conv1(x)))
        x = self.prelu(self.pool1(self.conv2(x)))
        x = self.prelu(self.pool1(self.conv3(x)))
        x = self.prelu(self.pool1(self.conv4(x)))
        x = x.view((batch_size, 256))
        bbox_layer = self.relue(self.linear0_0(x)).view((batch_size, -1))
        classification_layer = self.linear0_1(x).view((batch_size, -1))
        return {"bbox_pred": bbox_layer, "y_pred": classification_layer}