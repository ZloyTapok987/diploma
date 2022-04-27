import os
from io import StringIO

from gan_embed_to_svg_embed import ImageToVec, sample_all_glyphs
import torch
from deepsvg import utils
from deepsvg.difflib.tensor import SVGTensor
from deepsvg.svglib.utils import to_gif, make_grid, make_grid_lines, make_grid_grid, make_svg_text
from deepsvg.svglib.geom import Bbox
from deepsvg.svgtensor_dataset import SVGTensorDataset, load_dataset
from deepsvg.utils.utils import batchify, linear
import matplotlib.pyplot as plt
from deepsvg.svglib.svg import SVG
from configs.deepsvg.hierarchical_ordered_fonts import Config
from tokenizer import SvgTokenizer
from torch.nn.functional import normalize

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gan_to_svg_embed_path = "i2v"
deepsvg_pretrained_path = "pretrained/hierarchical_ordered_fonts.pth.tar"
#deepsvg_pretrained_path = "./checkpoints/finetuned8600.pth.tar"
GAN_EMBED_DIR = "logo_generation/gan_latent_spaces"
GAN_IMAGES_DIR = "logo_generation/gan_generated_logo"
result_dir = "logo_generation/result_logo"
text = "Wordmark"


model = ImageToVec()
model.load_state_dict(torch.load(gan_to_svg_embed_path))

cfg = Config()
deepsvg_model = cfg.make_model().to(device)
utils.load_model(deepsvg_pretrained_path, deepsvg_model)
deepsvg_model.eval()

glyph2label = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

long_glyphs = "bdfijlkt"
def make_svg_text_v2(text, svgs, to):
    tokenizer = SvgTokenizer()
    cur_width = 10
    cur_blank = 50
    res = tokenizer.empty_svg_tensor()
    for i in range(len(text)):
        if text[i] == " ":
            cur_width = cur_width + cur_blank
            continue
        svg = svgs[i]
        if text[i].isupper():
            max_width, max_height = tokenizer.get_max_width(svg, scale=50)
            tokenizer.tranlsate(svg, cur_width, 0)
            cur_width = cur_width + max_width + 4
        else:
            if text[i] in long_glyphs:
                max_width, max_height = tokenizer.get_max_width(svg, scale=30)
                tokenizer.tranlsate(svg, cur_width, 47-max_height)
            elif text[i] == 'm':
                max_width, max_height = tokenizer.get_max_width(svg, scale=25)
                tokenizer.tranlsate(svg, cur_width, 45 - max_height)
            else:
                max_width, max_height = tokenizer.get_max_width(svg, scale=20)
                tokenizer.tranlsate(svg, cur_width, 45 - max_height)
            cur_width = cur_width + max_width + 9
        res = tokenizer.concat_svg_tensors(res, svg)
    tokenizer.saveSvg(res, filename=to)

def draw_text(text, modell, z):
    svgs = []
    tokenizer = SvgTokenizer()
    for word in text:
        if word == " ":
            continue
        label_id = glyph2label.index(word)
        label, = batchify((torch.tensor(label_id),), device=device)
        commands_y, args_y = modell.greedy_sample(None, None, None, None, label=label, z=z)
        tensor_pred = SVGTensor.from_cmd_args(commands_y[0].cpu(), args_y[0].cpu())

        svg = SVG.from_tensor(tensor_pred.data, viewbox=Bbox(256),
                                    allow_empty=True).normalize().split_paths()
        #tensor = tokenizer.parseSvg(StringIO(svg))
        svgs.append(svg)

    return make_svg_text(text, svgs)


def load_gan_embeds():
    res = {}
    for file in os.listdir(GAN_EMBED_DIR):
        filename = os.path.join(GAN_EMBED_DIR, file)
        tmp = torch.load(filename, map_location=torch.device('cpu'))
        for k, v in tmp.items():
            # wordmark = k.split('/')[-1].split('.')[0].lower()
            # res[wordmark] = {}
            l = v['latent'].view(-1)
            arr = v['noise']
            for i in range(len(arr)):
                arr[i] = arr[i].view(-1)
            r = torch.cat(arr, dim=0)
            res[k] = (torch.cat([l, r], dim=0)  .view(-1))
    return res

def get_z(temperature=.3):
    z = torch.randn(1, 1, 1, cfg.model_cfg.dim_z).to(device) * temperature
    return z

gan_embeds = load_gan_embeds()

inp = torch.stack(list(gan_embeds.values()), dim=0)
deepSVG_embeds = model(inp)
print("calc deepSVG embeds")


tokenizer = SvgTokenizer()
svg_texts = []
for z in deepSVG_embeds:
    normalize(z, dim=0)
    svg = draw_text(text, deepsvg_model, z)
    #plt.imshow(svg.draw(return_png=True))
   # plt.show()
    tensor = tokenizer.parseSvg(StringIO(svg.to_str()))
    svg_texts.append(tensor)

print("convert it to svg tensors")
i = 0
for file in os.listdir(GAN_IMAGES_DIR):
    filename = os.path.join(GAN_IMAGES_DIR, file)
    logomark = tokenizer.parseSvg(filename, normalize=False)
    tokenizer.cut(logomark, 0, 128)
    #tokenizer.scale(logomark, 96)
    tokenizer.saveSvg(logomark)
    tokenizer.scale(svg_texts[i], 950)
    tokenizer.tranlsate(svg_texts[i], 130, 25)
    res = tokenizer.concat_svg_tensors(logomark, svg_texts[i])
    tokenizer.saveSvg(res, filename=os.path.join(result_dir, file))
    i = i + 1
print("done")



