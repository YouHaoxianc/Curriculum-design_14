import torch
from torchvision import transforms
from thop import profile
from yacs.config import CfgNode
from PIL import Image
import os
import matplotlib.pyplot as plt

from models import SPDCN

device = 'cpu'
config = CfgNode(dict(
    FACTOR = 128,
    resume = 'spdcn.pth', # the path to checkpoints
    norm_mean = [0.56347245, 0.50660025, 0.45908741],
    norm_std = [0.28393339, 0.2804536 , 0.30424776]
))

img_trans = transforms.Compose([
    transforms.Resize(384),
    transforms.ToTensor(),
    transforms.Normalize(mean = config.norm_mean, std = config.norm_std)
])
model = SPDCN(config)

checkpoint = torch.load(config.resume, map_location='cpu')
msg = model.load_state_dict(checkpoint, strict=False)

model = model.to(device)
impath = os.path.join("md-files", "5.jpg")

img_pil = Image.open(impath)
plt.imshow(img_pil)

image = img_trans(img_pil)[None, ...].to(device)
box = [
    [[532,  99], [540, 112]],
    [[329, 126], [334, 136]],
    [[255, 142], [268, 159]]
]

box = torch.cat((
    torch.zeros((len(box), 1)),
    torch.tensor(box).view(-1, 4),
), dim=1).to(device)
model.eval()
with torch.inference_mode():
    macs, params = profile(model, inputs=(image, box))
print(f'macs:{macs*1e-9},params:{params*1e-6}')