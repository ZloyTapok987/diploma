import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.deepsvg.hierarchical_ordered_fonts import Config
from deepsvg import utils
from deepsvg.difflib.tensor import SVGTensor
from deepsvg.svg_dataset import load_dataset
from deepsvg.svglib.geom import Bbox
from deepsvg.svglib.svg import SVG
from deepsvg.svgtensor_dataset import SVGFinetuneDataset
from deepsvg.utils import TrainVars
from deepsvg.utils.utils import batchify

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pretrained_path = "checkpoints/finetuned3100.pth.tar"

cfg = Config()
cfg.model_cfg.dropout = 0.  # for faster convergence
model = cfg.make_model().to(device)
model.eval()

dataset = load_dataset(cfg)

glyph2label = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def load_fintune_svgs(svg_dir):
    svgs = []
    labels = []
    for file in os.listdir(svg_dir):
        splited = file.split("_")
        if len(splited) < 2:
            continue

        print(file)
        filename = os.path.join(svg_dir, file)
        svg = SVG.load_svg(filename)
        svg.fill_(False)
        svg.normalize().zoom(0.9)
        svg.canonicalize()
        svg.split_paths()
        svg = svg.simplify_heuristic(tolerance=1)
        #plt.imshow(svg.draw(return_png=True))
        #plt.show()
        svg.filename = filename
        svgs.append(svg)

        label = splited[1][0]
        try:
            label_id = glyph2label.index(label)
            label, = batchify((torch.tensor(label_id),), device=device)
            labels.append(label)
        except:
            print(label)
    return svgs, labels

def sample_class(label, z=None, temperature=.3, filename=None, do_display=True, return_svg=False, return_png=False,
                 *args, **kwargs):
    label_id = glyph2label.index(label)

    if z is None:
        z = torch.randn(1, 1, 1, cfg.model_cfg.dim_z).to(device) * temperature

    label, = batchify((torch.tensor(label_id),), device=device)
    commands_y, args_y = model.greedy_sample(None, None, None, None, label=label, z=z)
    tensor_pred = SVGTensor.from_cmd_args(commands_y[0].cpu(), args_y[0].cpu()).drop_sos().unpad()

    svg_path_sample = SVG.from_tensor(tensor_pred.data, viewbox=Bbox(256), allow_empty=True).normalize().split_paths()

    if return_svg:
        return svg_path_sample

    return svg_path_sample.draw(file_path=filename, do_display=do_display, return_png=return_png, *args, **kwargs)



def finetune_model(nb_augmentations=1500):
    svgs, labels = load_fintune_svgs("dataset/dataset/letter")

    utils.load_model(pretrained_path, model)
    finetune_dataset = SVGFinetuneDataset(dataset, svgs, labels, frac=1, nb_augmentations=nb_augmentations)
    dataloader = DataLoader(finetune_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False,
                            num_workers=cfg.loader_num_workers, collate_fn=cfg.collate_fn)


    train_vars = TrainVars()
    cfg.set_train_vars(train_vars, dataloader)

    # Optimizer, lr & warmup schedulers
    optimizers = cfg.make_optimizers(model)
    scheduler_lrs = cfg.make_schedulers(optimizers, epoch_size=len(dataloader))
    scheduler_warmups = cfg.make_warmup_schedulers(optimizers, scheduler_lrs)

    loss_fns = [l.to(device) for l in cfg.make_losses()]
    checkpoint_dir = "checkpoints"
    epoch = 0

    for step, data in enumerate(dataloader):
        model.train()
        model_args = [data[arg].to(device) for arg in cfg.model_args]

        data["label"] = data["label"].view(-1)
        labels = data["label"].to(device).view(-1) if "label" in data else None
        params_dict, weights_dict = cfg.get_params(step, epoch), cfg.get_weights(step, epoch)

        for i, (loss_fn, optimizer, scheduler_lr, scheduler_warmup, optimizer_start) in enumerate(
                zip(loss_fns, optimizers, scheduler_lrs, scheduler_warmups, cfg.optimizer_starts), 1):
            optimizer.zero_grad()

            output = model(*model_args, params=params_dict)
            loss_dict = loss_fn(output, labels, weights=weights_dict)

            loss_dict["loss"].backward()
            if cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            optimizer.step()
            if scheduler_lr is not None:
                scheduler_lr.step()
            if scheduler_warmup is not None:
                scheduler_warmup.step()

            if step % 100 == 0:
                print(f"Step {step}: loss: {loss_dict['loss']}")
                svgs = model(*model_args)
                with torch.no_grad():
                    # Visualization
                    output = None
                    cfg.visualize(model, output, train_vars, step, epoch, None, "visualize")

                #plt.show()
                utils.save_ckpt_list(checkpoint_dir, model, cfg, optimizers, scheduler_lrs, scheduler_warmups, step=step)

    print("Finetuning done.")

if __name__ == '__main__':
    finetune_model()
