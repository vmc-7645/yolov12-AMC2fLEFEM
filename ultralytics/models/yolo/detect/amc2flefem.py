import torch
import torch.nn as nn
import torch.nn.functional as F

class LEF(nn.Module):
    def __init__(self):
        super(LEF, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)

    def forward(self, x):
        p1 = self.pool1(x)
        p2 = F.interpolate(self.pool2(x), size=x.shape[2:], mode='bilinear', align_corners=False)
        p3 = F.interpolate(self.pool3(x), size=x.shape[2:], mode='bilinear', align_corners=False)
        p4 = F.interpolate(self.pool4(x), size=x.shape[2:], mode='bilinear', align_corners=False)
        combined = torch.cat([p1, p2, p3, p4], dim=1)
        return combined

class FEM(nn.Module):
    def __init__(self, in_channels):
        super(FEM, self).__init__()
        self.conv = nn.Conv2d(in_channels * 5, in_channels, kernel_size=1)

    def forward(self, x, lef_output):
        fused = torch.cat([x, lef_output], dim=1)
        fused = self.conv(fused)
        return fused

class SimAM(nn.Module):
    def __init__(self, lambda_val=1e-4):
        super(SimAM, self).__init__()
        self.lambda_val = lambda_val

    def forward(self, x):
        n = x.shape[2] * x.shape[3] - 1
        d = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        v = d.sum(dim=[2, 3], keepdim=True) / n
        E_inv = d / (4 * (v + self.lambda_val)) + 0.5
        return x * torch.sigmoid(E_inv)

class C2fLEFEM(nn.Module):
    def __init__(self, in_channels):
        super(C2fLEFEM, self).__init__()
        self.c2f = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.lef = LEF()
        self.fem = FEM(in_channels)

    def forward(self, x):
        c2f_out = self.c2f(x)
        lef_out = self.lef(c2f_out)
        fem_out = self.fem(c2f_out, lef_out)
        return fem_out

class AMC2fLEFEM(nn.Module):
    def __init__(self, in_channels):
        super(AMC2fLEFEM, self).__init__()
        self.c2flefem = C2fLEFEM(in_channels)
        self.simam = SimAM()

    def forward(self, x):
        c2flefem_out = self.c2flefem(x)
        amc2flefem_out = self.simam(c2flefem_out)
        return amc2flefem_out