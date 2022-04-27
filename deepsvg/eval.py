import os
# os.chdir("..")

from deepsvg.svglib.svg import SVG

from deepsvg import utils
from deepsvg.difflib.tensor import SVGTensor
from deepsvg.svglib.utils import to_gif, make_grid, make_grid_lines, make_grid_grid
from deepsvg.svglib.geom import Bbox
from deepsvg.svgtensor_dataset import SVGTensorDataset, load_dataset
from deepsvg.utils.utils import batchify, linear
import matplotlib.pyplot as plt

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#pretrained_path = "./checkpoints/finetuned8600.pth.tar"
pretrained_path = "pretrained/hierarchical_ordered_fonts.pth.tar"
from configs.deepsvg.hierarchical_ordered_fonts import Config

cfg = Config()
model = cfg.make_model().to(device)
utils.load_model(pretrained_path, model)
model.eval()

#dataset = load_dataset(cfg)

glyph2label = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def sample_class(label, z=None, temperature=.3, filename=None, do_display=True, return_svg=False, return_png=False,
                 *args, **kwargs):
    label_id = glyph2label.index(label)

    if z is None:
        z = torch.randn(1, 1, 1, cfg.model_cfg.dim_z).to(device) * temperature

    label, = batchify((torch.tensor(label_id),), device=device)
    commands_y, args_y = model.greedy_sample(None, None, None, None, label=label, z=z)

    tensor_pred = SVGTensor.from_cmd_args(commands_y[0].cpu(), args_y[0].cpu())

    svg_path_sample = SVG.from_tensor(tensor_pred.data, viewbox=Bbox(256), allow_empty=True).normalize().split_paths()

    if return_svg:
        return svg_path_sample

    return svg_path_sample.draw(file_path=filename, do_display=do_display, return_png=return_png, *args, **kwargs)


def easein_easeout(t):
    return t * t / (2. * (t * t - t) + 1.)


def interpolate(z1, z2, label, n=25, filename=None, ease=True, do_display=True):
    alphas = torch.linspace(0., 1., n)
    if ease:
        alphas = easein_easeout(alphas)
    z_list = [(1 - a) * z1 + a * z2 for a in alphas]

    img_list = [sample_class(label, z, do_display=False, return_png=True) for z in z_list]
    to_gif(img_list + img_list[::-1], file_path=filename, frame_duration=1 / 12)


PAD_VAL = -1

MAX_NUM_GROUPS = cfg.max_num_groups # Number of paths (N_P)
MAX_SEQ_LEN = cfg.max_seq_len  # Number of commands (N_C)
MAX_TOTAL_LEN = cfg.max_total_len


def get_data(t_sep, fillings, model_args=None, label = None):
    res = {}

    if model_args is None:
        model_args = model_args

    pad_len = max(MAX_NUM_GROUPS - len(t_sep), 0)

    t_sep.extend([torch.empty(0, 14)] * pad_len)
    fillings.extend([0] * pad_len)

    t_grouped = [SVGTensor.from_data(torch.cat(t_sep, dim=0), PAD_VAL=PAD_VAL).add_eos().add_sos().pad(
        seq_len=MAX_TOTAL_LEN + 2)]
    need = SVGTensor.from_data(torch.cat(t_sep, dim=0), PAD_VAL=PAD_VAL).pad(
        seq_len=MAX_TOTAL_LEN + 2)
    t_sep = [SVGTensor.from_data(t, PAD_VAL=PAD_VAL, filling=f).add_eos().add_sos().pad(
        seq_len=MAX_SEQ_LEN + 2) for
        t, f in zip(t_sep, fillings)]

    # plt.imshow(SVG.from_tensor(t_grouped[0].copy().drop_sos().unpad().data).draw(return_png=True))
    # plt.show()

    for arg in set(model_args):
        if "_grouped" in arg:
            arg_ = arg.split("_grouped")[0]
            t_list = t_grouped
        else:
            arg_ = arg
            t_list = t_sep

        if arg_ == "tensor":
            res[arg] = t_list

        if arg_ == "commands":
            res[arg] = torch.stack([t.cmds() for t in t_list])

        if arg_ == "args_rel":
            res[arg] = torch.stack([t.get_relative_args() for t in t_list])
        if arg_ == "args":
            res[arg] = torch.stack([t.args() for t in t_list])

    if "filling" in model_args:
        res["filling"] = torch.stack([torch.tensor(t.filling) for t in t_sep]).unsqueeze(-1)

    if "label" in model_args:
        res["label"] = label

    if id is not None:
        res["id"] = id

    return res

def get_all_alphabet(svg_dir, file_pref, filename=None):
    zs = []
    for file in os.listdir(svg_dir):
        if not file.startswith(file_pref):
            continue

        if len(file.split("_")) == 1:
            continue

        pre_label = file.split("_")[1][0]
        label = torch.tensor(glyph2label.index(pre_label))

        svg = SVG.load_svg(os.path.join(svg_dir, file))
        svg.fill_(False)
        svg.normalize().zoom(0.9)
        svg.canonicalize()
        svg.split_paths()
        svg = svg.simplify_heuristic(tolerance=1)

        t_sep, fillings = svg.to_tensor(concat_groups=False, PAD_VAL=-1), svg.to_fillings()

        data = get_data(t_sep, fillings, model_args=cfg.model_args, label=label)

        model_args = batchify((data[key] for key in cfg.model_args), device)

        with torch.no_grad():
            z = model(*model_args, encode_mode=True)
            zs.append(z)
            # sample_class("v", z)
           # plt.imshow(sample_class('c', z, with_points=True, return_png=True, with_handles=True, with_moves=False))
            #plt.show()
    if len(zs) == 0:
        return None

    res = zs[0]

    for i in range(len(zs)):
        if i == 0:
            continue
        res = res + zs[i]

    res = res / float(len(zs))

    if filename is not None:
        sample_all_glyphs(res, filename)

    return res


def mega_fail(filename):
    svg = SVG.load_svg(filename)
    svg.fill_(False)
    svg.normalize().zoom(0.9)
    svg.canonicalize()
    svg.split_paths()
    svg = svg.simplify_heuristic(tolerance=1)

    t_sep, fillings = svg.to_tensor(concat_groups=False, PAD_VAL=-1), svg.to_fillings()

    letter = filename.split("_")[1][0]

    data = get_data(t_sep, fillings, model_args=cfg.model_args, label=torch.tensor(glyph2label.index(letter)))

    model_args = batchify((data[key] for key in cfg.model_args), device)

    with torch.no_grad():
        z = model(*model_args, encode_mode=True)
        #sample_class("v", z)
        plt.imshow(sample_class(letter, z, with_points=True, return_png=True, with_handles=True, with_moves=False))
        plt.show()
    sample_all_glyphs(z, filename="lol.svg")


def encode_icon(idx):
    data = dataset.get(id=idx, random_aug=False)
    model_args = batchify((data[key] for key in cfg.model_args), device)
    with torch.no_grad():
        z = model(*model_args, encode_mode=True)
    return z


def interpolate_icons(idx1, idx2, label, n=25, *args, **kwargs):
    z1, z2 = encode_icon(idx1), encode_icon(idx2)
    interpolate(z1, z2, label, n=n, *args, **kwargs)


def get_z(temperature=.3):
    z = torch.randn(1, 1, 1, cfg.model_cfg.dim_z).to(device) * temperature
    return z


def sample_all_glyphs(z, filename=None):
    #svg_digits = [sample_class(glyph, z=z, return_svg=True) for glyph in "0123456789"]
    svg_lower = [sample_class(glyph, z=z, return_svg=True) for glyph in "abcdefghijklmnopqrstuvwxyz"]
    svg_upper = [sample_class(glyph, z=z, return_svg=True) for glyph in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]

    grid = make_grid_lines([svg_lower, svg_upper])
    grid.draw(file_path=filename)


#for c in glyph2label:
#    for i in range(100):
#        z = get_z()
#        plt.imshow(sample_class('q', z=z, with_points=True, return_png=True, with_handles=True, with_moves=False))
#        plt.show()

#plt.imshow(sample_class('p', z=get_z(), with_points=True, return_png=True, with_handles=True, with_moves=False))
#plt.show()
if False:
    res = {}
    svg_letter_dir = "dataset/dataset/letter"
    svg_out_dir = "deepsvg_latent_spaces"

    exit()
    #mega_fail("dataset/dataset/letter/Adobe_A.svg")
    #get_all_alphabet("dataset/dataset/letter", "trendyol", "lol.svg")
    #exit()

    for file in os.listdir(svg_letter_dir):
        #print(file)
        wordmark = file.split("_")[0].lower()
        if wordmark in res:
            continue
        z = get_all_alphabet(svg_letter_dir, wordmark,"z_sample/" + wordmark + ".svg")
        if z is None:
            continue


        print(wordmark)
        res[wordmark] = z
        torch.save(res, os.path.join(svg_out_dir, wordmark + ".pt"))

#mega_fail("a")

# sample_all_glyphs(z)

# z1, z2 = get_z(), get_z()
# interpolate(z1, z2, "9")
