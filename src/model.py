import torch
import torch.nn as nn

def center_crop_to(tensor, target_tensor):
    """Crop `tensor` do spatial rozmiaru `target_tensor`, zachowując centrowanie."""
    _, _, h, w, d = target_tensor.shape
    _, _, H, W, D = tensor.shape
    dh, dw, dd = (H - h) // 2, (W - w) // 2, (D - d) // 2
    return tensor[:, :, dh:dh+h, dw:dw+w, dd:dd+d]


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super(UNet3D, self).__init__()
        self.encoder1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool3d(2)
        self.encoder2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool3d(2)
        self.encoder3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool3d(2)
        self.encoder4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool3d(2)
        self.bottleneck = DoubleConv(256, 512)
        self.upconv4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(512, 256)
        self.upconv3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(256, 128)
        self.upconv2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(128, 64)
        self.upconv1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(64, 32)
        self.out_conv = nn.Conv3d(32, out_channels, kernel_size=1)
        self.conv_final = nn.Conv3d(in_channels=32, out_channels=1, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder
        dec4 = self.upconv4(bottleneck)
        enc4 = center_crop_to(enc4, dec4)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        enc3 = center_crop_to(enc3, dec3)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        enc2 = center_crop_to(enc2, dec2)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        enc1 = center_crop_to(enc1, dec1)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.conv_final(dec1))
