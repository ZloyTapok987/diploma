import random
from random import randrange
from StyleLogo.model import Generator
from torchvision import utils
from tqdm import tqdm
import torch
import os

import clip
import numpy as np
import PIL.Image
import torch

from tokenizer import SvgTokenizer

from StyleCLIP.embedding import get_delta_t
from StyleCLIP.manipulator import Manipulator
from StyleCLIP.mapper import get_delta_s
from StyleCLIP.wrapper import Generator

from gan_embed_to_svg_embed import ImageToVec
import subprocess
from generate_logo_deepvecfont import drawSvgTextTo


NUM_LOGOS = 15192

def prepare():
    device = torch.device('cuda:0')
    # pretrained ffhq generator
    ckpt = '/content/drive/MyDrive/diploma/out.pkl'  # 'checkpoint/out.pkl'
    G = Generator(ckpt, device)
    # CLIP
    clipModel, preprocess = clip.load("ViT-B/32", device=device)
    # global image direction
    fs3 = np.load('/content/drive/MyDrive/diploma/StyleCLIP/tensor/fs3logo.npy')  # 'StyleCLIP/tensor/fs3logo.npy')
    manipulator = Manipulator(G, device, face_preprocess=False, dataset_name="logo", num_images=1999)

    i2v = ImageToVec()
    i2v.load_state_dict(torch.load("checkpoint/i2v"))

    return manipulator, clipModel, fs3, i2v

manipulator, clipModel, fs3, i2v = prepare()

def generate(sample, latent, pics, truncation, g_ema, device, mean_latent):

    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(pics)):
            sample_z = torch.randn(sample, latent, device=device)

            sample, _ = g_ema(
                [sample_z], truncation=truncation, truncation_latent=mean_latent
            )

            utils.save_image(
                sample,
                f"sample/{str(i).zfill(6)}.png",
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )


def generate_logomark(manipulator, clipModel, fs3, target):
    neutral = 'logo'
    beta_threshold = 0.10
    lst_alpha = [-2]
    manipulator.set_alpha(lst_alpha)

    classnames = [neutral, target]
    # get delta_t in CLIP text space
    delta_t = get_delta_t(classnames, clipModel)
    # get delta_s in global image directions and text directions that satisfy beta threshold
    delta_s, num_channel = get_delta_s(fs3, delta_t, manipulator, beta_threshold=beta_threshold)

    styles = manipulator.manipulate(delta_s)

    idx = randrange(1999)
    all_imgs = manipulator.synthesis_from_styles(styles, idx, idx + 1)

    H, W = (128, 128)
    img_arr = (all_imgs[0].permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).numpy()[0]
    img = PIL.Image.fromarray(img_arr, 'RGB')
    img = img.resize((H, W), PIL.Image.LANCZOS)
    return img


def vectorize_logomark(img_path):
    bashCommand = "python diffvg/apps/painterly_rendering.py {} --num_paths 64 --use_blob --max_width 4".format(img_path)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    tokenizer = SvgTokenizer()
    base = os.path.basename(img_path)
    return tokenizer.parseSvg("diffvg/apps/results/{}".format(base))


def generate_wordmark(img, i2v, text):
    img.save("logomark.jpg")
    bashCommand = "python StyleLogo/projector.py --size 128 --w_plus --ckpt checkpoint/175000.pt /content/drive/MyDrive/diploma/sample/000029.png"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    img_projection = torch.load("projection/logomark.pt")
    vec_embed = i2v([img_projection])
    drawSvgTextTo(text, "wordmark.svg", z=vec_embed)
    tokenizer = SvgTokenizer()
    return tokenizer.parseSvg("wordmark.svg")


def concat_right(logomark, wordmark):
    tokenizer = SvgTokenizer()
    tokenizer.cut(logomark, 0, 128)
    # tokenizer.scale(logomark, 96)
    tokenizer.saveSvg(logomark)
    tokenizer.scale(wordmark, 950)
    tokenizer.tranlsate(wordmark, 130, 25)
    res = tokenizer.concat_svg_tensors(logomark, wordmark)
    return res

def concat_below(logomark, wordmark):
    tokenizer = SvgTokenizer()
    tokenizer.cut(logomark, 0, 128)
    # tokenizer.scale(logomark, 96)
    tokenizer.saveSvg(logomark)
    tokenizer.scale(wordmark, 100)
    tokenizer.tranlsate(wordmark, 32, -140)
    res = tokenizer.concat_svg_tensors(logomark, wordmark)
    return res

def concat_logomark_wordmark(logomark, wordmark):
    if True:
        concat_right(logomark, wordmark)
    else:
        concat_below(logomark, wordmark)

def generate_full_logo(company_name, target_text):
    tokenizer = SvgTokenizer()
    logomark = generate_logomark(manipulator, clipModel, fs3, target_text)
    wordmark = generate_wordmark(logomark, i2v, company_name)
    logomark_vec = vectorize_logomark("logomark.jpg")
    return concat_logomark_wordmark(logomark_vec, wordmark)


tokenizer = SvgTokenizer()
res = generate_full_logo("Wordmark", "red logo")
tokenizer.saveSvg(res, filename="result.svg")
