import os

import torch
import numpy as np
from matplotlib import pyplot as plt
import cv2
from PIL import Image
from fire import Fire
from scripts.srcnnSR import SrcnnSR
from torchvision import transforms
from pathlib import Path
import tqdm
from scripts.dataloader import DIV2KDataset, Set5Dataset, Set14Dataset
from skimage.metrics import structural_similarity


def main(config="test"):
    scale_factor = 4

    app = SrcnnSR(config)
    app.load()

    best_model = app.model
    best_model.eval()

    device = app.device
    test_dir = "tests" + f'\\X{scale_factor}' + "\\" + app.getModelName() + "\\Set14"

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    dataset = Set14Dataset(
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
        lr_scale=scale_factor
    )

    psnr_list = list()
    score_list = list()
    bicubic_psnr_list = list()
    names = ['baby', 'bird', 'butterfly', 'head', 'woman']
    idx = 0
    for img, label in dataset:
        bicubic_test_img = np.clip(torch.squeeze(img).cpu().detach().numpy(), 0, 1)
        bicubic_test_img = ((bicubic_test_img * 255) / np.max(bicubic_test_img)).astype(np.uint8)

        label = np.clip(torch.squeeze(label).cpu().detach().numpy(), 0, 1)
        label = ((label * 255) / np.max(label)).astype(np.uint8)
        bicubic_psnr = cv2.PSNR(label, bicubic_test_img)
        bicubic_psnr_list.append(bicubic_psnr)

        x = torch.unsqueeze(img,0)
        x = x.to(device)

        output = best_model(x)
        output = np.clip(torch.squeeze(output).cpu().detach().numpy(), 0, 1)
        output = ((output * 255) / np.max(output)).astype(np.uint8)
        psnr = cv2.PSNR(output, label)
        (score, diff) = structural_similarity(output, label, full=True, channel_axis=0)
        psnr_list.append(psnr)
        score_list.append(score)
        print(f"Wartosc PSNR dla obrazu {names[idx]}, wynosi: {psnr}")
        img = Image.fromarray(np.moveaxis(output, 0, -1), 'RGB')
        img.save(test_dir + '\\' + names[idx] + '.png')
        idx=idx+1

    print(f"Mean PSNR: {np.mean(psnr_list)}, mean SSIM: {np.mean(score_list)}, mean bicubic PSNR:"
          f"{np.mean(bicubic_psnr_list)}")

    #print(f"Mean PSNR: {np.mean(psnr_list)}, mean SSIM: {np.mean(score_list)}")

    """
    for file in tqdm.tqdm(os.listdir(images_path)):
        img = Image.open(images_path + '\\' + file)
        width = img.size[0] * scale_factor
        height = img.size[1] * scale_factor
        dim = (width, height)
        img = np.array(img)#[:, :, 0]
        image = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
        convert_tensor = transforms.ToTensor()
        x = torch.unsqueeze(convert_tensor(image),0)
        x = x.to(device)

        output = best_model(x)
        output = np.clip(torch.squeeze(output).cpu().detach().numpy(), 0, 1)
        output = ((output * 255) / np.max(output)).astype(np.uint8)
        img = Image.fromarray(np.moveaxis(output, 0, -1), 'RGB')
        img.save(test_dir + '\\' + file)
    """

if __name__ == '__main__':
    Fire(main)