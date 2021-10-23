from torch import nn
from efficientnet_pytorch import EfficientNet

class EfficientNetAll(nn.Module):
    def __init__(self, mode, advprop, num_classes=1000, feature = 1280, ArcFeature = 512):
        super(EfficientNetAll, self).__init__()
        self.efficientNet = EfficientNet.from_pretrained(mode, advprop=advprop)
        self.dropout = nn.Dropout(0.5)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(feature, ArcFeature)  # B0:1280, B5:2048, B6:2304
        self.bn2 = nn.BatchNorm1d(ArcFeature)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(ArcFeature, num_classes)

    def forward(self, x):
        self.efficientNet(x)
        features = self.efficientNet.extract_features(x)
        x = self.avgpool(features)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        embedding = self.fc(x)
        x = self.bn2(embedding)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x, embedding


class classifier(nn.Module):
    def __init__(self, num_classes=1000, ArcFeature = 512):
        super(classifier, self).__init__()
        self.fc = nn.Linear(ArcFeature, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x