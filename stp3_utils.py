import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18
import torch.nn.functional as F
from stp3_convolutions import UpsamplingAdd
     
class VehicleDecoder(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.n_classes = n_classes
        backbone = resnet18(pretrained=False, zero_init_residual=True)

        self.first_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        shared_out_channels = in_channels
        self.up3_skip = UpsamplingAdd(256, 128, scale_factor=2)
        self.up2_skip = UpsamplingAdd(128, 64, scale_factor=2)
        self.up1_skip = UpsamplingAdd(64, shared_out_channels, scale_factor=2)

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, self.n_classes, kernel_size=1, padding=0),
        )
    def forward(self, x):
        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)
        # (H, W)
        skip_x = {'1': x}

        # (H/2, W/2)
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        skip_x['2'] = x

        # (H/4 , W/4)
        x = self.layer2(x)
        skip_x['3'] = x

        # (H/8, W/8)
        x = self.layer3(x)  # (b*s, 256, 25, 25)

        # First upsample to (H/4, W/4)
        x = self.up3_skip(x, skip_x['3'])

        # Second upsample to (H/2, W/2)
        x = self.up2_skip(x, skip_x['2'])

        # Third upsample to (H, W)
        x = self.up1_skip(x, skip_x['1'])
        
        segmentation_output = self.segmentation_head(x)

        return segmentation_output.view(b, s, *segmentation_output.shape[1:])

class PedestrianDecoder(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.n_classes = n_classes
        backbone = resnet18(pretrained=False, zero_init_residual=True)

        self.first_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        shared_out_channels = in_channels
        self.up3_skip = UpsamplingAdd(256, 128, scale_factor=2)
        self.up2_skip = UpsamplingAdd(128, 64, scale_factor=2)
        self.up1_skip = UpsamplingAdd(64, shared_out_channels, scale_factor=2)

        self.pedestrian_head = nn.Sequential(
                nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(shared_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_out_channels, self.n_classes, kernel_size=1, padding=0),
            )

    def forward(self, x):
        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)
        # (H, W)
        skip_x = {'1': x}

        # (H/2, W/2)
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        skip_x['2'] = x

        # (H/4 , W/4)
        x = self.layer2(x)
        skip_x['3'] = x

        # (H/8, W/8)
        x = self.layer3(x)  # (b*s, 256, 25, 25)

        # First upsample to (H/4, W/4)
        x = self.up3_skip(x, skip_x['3'])

        # Second upsample to (H/2, W/2)
        x = self.up2_skip(x, skip_x['2'])

        # Third upsample to (H, W)
        x = self.up1_skip(x, skip_x['1'])

        pedestrian_output = self.pedestrian_head(x)
      
        return pedestrian_output.view(b, s, *pedestrian_output.shape[1:])

class SegmentationLoss(nn.Module):
    def __init__(self, class_weights, ignore_index=255, use_top_k=False, top_k_ratio=1.0, future_discount=1.0):
        super().__init__()
        self.class_weights = class_weights
        self.ignore_index = ignore_index
        self.use_top_k = use_top_k
        self.top_k_ratio = top_k_ratio
        self.future_discount = future_discount

    def forward(self, prediction, target, n_present=1):
        if target.shape[-3] != 1:
            raise ValueError('segmentation label must be an index-label with channel dimension = 1.')
        b, s, c, h, w = prediction.shape

        prediction = prediction.view(b * s, c, h, w)
        target = target.view(b * s, h, w)
        loss = F.cross_entropy(
            prediction,
            target,
            ignore_index=self.ignore_index,
            reduction='none',
            weight=self.class_weights.to(target.device),
        )

        loss = loss.view(b, s, h, w)

        assert s >= n_present
        future_len = s - n_present
        future_discounts = self.future_discount ** torch.arange(1, future_len+1, device=loss.device, dtype=loss.dtype)
        discounts = torch.cat([torch.ones(n_present, device=loss.device, dtype=loss.dtype), future_discounts], dim=0)
        discounts = discounts.view(1, s, 1, 1)
        loss = loss * discounts

        loss = loss.view(b, s, -1)
        if self.use_top_k:
            # Penalises the top-k hardest pixels
            k = int(self.top_k_ratio * loss.shape[2])
            loss, _ = torch.sort(loss, dim=2, descending=True)
            loss = loss[:, :, :k]

        return torch.mean(loss)