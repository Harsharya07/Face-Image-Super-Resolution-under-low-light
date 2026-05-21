#!/usr/bin/env python3
import torch, torch.nn as nn, torch.nn.functional as F
import cv2, numpy as np
import sys
import types

import torchvision.transforms.functional as TF
functional_tensor = types.ModuleType('torchvision.transforms.functional_tensor')
functional_tensor.rgb_to_grayscale = TF.rgb_to_grayscale
sys.modules['torchvision.transforms.functional_tensor'] = functional_tensor

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3,1,1,bias=False), nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.1,True),
            nn.Conv2d(out_ch,out_ch,3,1,1,bias=False), nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.1,True),
        )
    def forward(self, x): return self.net(x)

class LightEnhancerUNet(nn.Module):
    def __init__(self, ch=(32,64,128,256), scale=0.65):
        super().__init__()
        c=ch; self.scale=scale
        self.enc1=DoubleConv(3,c[0]); self.enc2=DoubleConv(c[0],c[1])
        self.enc3=DoubleConv(c[1],c[2]); self.enc4=DoubleConv(c[2],c[3])
        self.bottleneck=DoubleConv(c[3],c[3])
        self.dec4=DoubleConv(c[3]*2,c[2]); self.dec3=DoubleConv(c[2]*2,c[1])
        self.dec2=DoubleConv(c[1]*2,c[0]); self.dec1=DoubleConv(c[0]*2,c[0])
        self.pool=nn.MaxPool2d(2)
        self.up=nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False)
        self.head=nn.Sequential(nn.Conv2d(c[0],16,3,1,1),nn.LeakyReLU(0.1,True),nn.Conv2d(16,3,1),nn.Tanh())
    def forward(self,x):
        e1=self.enc1(x); e2=self.enc2(self.pool(e1))
        e3=self.enc3(self.pool(e2)); e4=self.enc4(self.pool(e3))
        b=self.bottleneck(self.pool(e4))
        d4=self.dec4(torch.cat([self.up(b),e4],1))
        d3=self.dec3(torch.cat([self.up(d4),e3],1))
        d2=self.dec2(torch.cat([self.up(d3),e2],1))
        d1=self.dec1(torch.cat([self.up(d2),e1],1))
        return (x+self.head(d1)*self.scale).clamp(0,1)

class LLFSRPipeline:
    """Drop-in pipeline for LLFSR-Net v5."""
    LR = 128
    HR = 512

    def __init__(self, enhancer_ckpt, gfpgan_ckpt, fidelity_weight=0.5, device=None):
        self.w = fidelity_weight
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.enhancer = LightEnhancerUNet().to(self.device).eval()
        ck = torch.load(enhancer_ckpt, map_location=self.device, weights_only=False)
        self.enhancer.load_state_dict(ck["model_state"])

        from gfpgan import GFPGANer
        self.gfpganer = GFPGANer(
            model_path=gfpgan_ckpt, upscale=4, arch="clean",
            channel_multiplier=2, bg_upsampler=None
        )
        print(f"LLFSRPipeline ready on {self.device}")

    def enhance(self, img_bgr):
        h, w = img_bgr.shape[:2]; s = min(h, w)
        crop = img_bgr[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2]
        lr   = cv2.resize(crop, (self.LR, self.LR), cv2.INTER_AREA)

        lr_rgb = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(lr_rgb).permute(2,0,1).float().unsqueeze(0)/255.
        with torch.no_grad():
            enh = self.enhancer(t.to(self.device))
        enh_np = (enh[0].permute(1,2,0).cpu().numpy()*255).clip(0,255).astype(np.uint8)

        enh_bgr = cv2.cvtColor(enh_np, cv2.COLOR_RGB2BGR)
        _, _, restored = self.gfpganer.enhance(
            enh_bgr, has_aligned=False, paste_back=True, weight=self.w
        )
        return restored


if __name__ == "__main__":
    import sys, os
    if len(sys.argv) < 4:
        print("Usage: python ui_inference.py <input.jpg> <enhancer.pth> <GFPGANv1.4.pth>")
        sys.exit(1)

    pipeline = LLFSRPipeline(sys.argv[2], sys.argv[3])
    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Cannot read {sys.argv[1]}"); sys.exit(1)
    result = pipeline.enhance(img)
    out = os.path.splitext(sys.argv[1])[0] + "_LLFSR_v5.png"
    cv2.imwrite(out, result)
    print(f"Saved: {out}")
