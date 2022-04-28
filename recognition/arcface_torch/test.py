import argparse
from email.policy import strict

import cv2
import numpy as np
import torch

from backbones import get_model


@torch.no_grad()
def inference(weight, name, img=None):
    # if img is None:
    #     img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    # else:
    #     img = cv2.imread(img)
    #     img = cv2.resize(img, (112, 112))

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = np.transpose(img, (2, 0, 1))
    # img = torch.from_numpy(img).unsqueeze(0).float()
    # img.div_(255).sub_(0.5).div_(0.5)
    net = get_model(name, fp16=False)
    state_dict = torch.load(weight, map_location='cpu')
    net.load_state_dict(state_dict, strict=True)
    net.eval()
    # feat = net(img).numpy()
    print(net)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r100', help='backbone network')
    parser.add_argument('--weight', type=str, default='./weights/backbone.pth')
    # parser.add_argument('--weight', type=str, default='./weights/rank_0_softmax_weight.pt')
    parser.add_argument('--img', type=str, default=None)
    args = parser.parse_args()
    inference(args.weight, args.network, args.img)
