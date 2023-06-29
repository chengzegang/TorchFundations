from typing import Callable, Collection, Iterable, List, Optional, Sequence, Tuple
import torch
import torchdata.datapipes as dp
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
import torchvision.transforms.v2.functional as TF
from PIL import Image
from torch.nn import Identity
from torch.utils.tensorboard import SummaryWriter
from functools import partial, wraps
import os
from dataclasses import dataclass, field
import yaml
from torch.nn import Module
from ...meta import Available
from torch import Tensor
import matplotlib.pyplot as plt
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import LambdaLR
import math
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from tqdm.auto import tqdm
import typer

app = typer.Typer(pretty_exceptions_enable=False)


def cosine_lr(current: int, total: int, shift: int = 0, warmup: int = 0) -> float:
    current += shift
    if current < warmup:
        return current / (warmup + 1e-8)
    elif current >= total:
        return 1e-8
    else:
        factor = (current - warmup) / (total - warmup + 1e-8)
        factor = (math.cos(math.pi * factor) + 1) / 2
        return factor


def datapipe(
    root: str,
    exts: Collection[str],
    recursive: bool = False,
    absolute_path: bool = False,
    shuffle: bool = False,
    size: Tuple[int, int] = (256, 256),
    batch_size: int = 32,
    transform: Callable[[Tensor], Tensor] | None = None,
) -> dp.iter.IterDataPipe:
    transform = transform or Identity()
    return (
        dp.iter.FileLister(
            root,
            masks=[f"*{ex}" for ex in exts],
            recursive=recursive,
            abspath=absolute_path,
            non_deterministic=False,
        )
        .shuffle(buffer_size=0 if not shuffle else 10000)
        .sharding_filter()
        .map(Image.open)
        .map(partial(TF.resize_image_pil, size=size))
        .map(TF.convert_dtype)
        .map(transform)
        .batch(batch_size)
        .collate()
    )


def plot_train_results(
    ground_truth: Tensor, predicted: Tensor, step: int, epoch: int, loss: float
) -> plt.Figure:
    ground_truth = ground_truth.detach().cpu().permute(1, 2, 0)
    predicted = predicted.detach().cpu().permute(1, 2, 0).clamp(0, 1)
    fig, ax = plt.subplots(1, 2)
    fig.suptitle(f"Epoch: {epoch} Step: {step} Loss: {loss}")

    ax[0].imshow(ground_truth)
    ax[0].set_title("Ground Truth")
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    ax[1].imshow(predicted)
    ax[1].set_title("Predicted")
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    fig.tight_layout()
    return fig


def make_logdirs(root_log_dir: str, name: str) -> Tuple[str, str, str, str]:
    os.makedirs(root_log_dir, exist_ok=True)
    log_dir = os.path.join(root_log_dir, name)
    os.makedirs(log_dir, exist_ok=True)
    run_dir = os.path.join(log_dir, "runs")
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    out_dir = os.path.join(log_dir, "out")
    os.makedirs(out_dir, exist_ok=True)
    return log_dir, run_dir, checkpoint_dir, out_dir


def save_checkpoint(model: Module, checkpoint_dir: str) -> str:
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
    last_checkpoint_path = checkpoint_path
    model.eval()
    torch.save(model.state_dict(), checkpoint_path)
    return last_checkpoint_path


def save_params(kwargs_key: str = "param_dir") -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(**kwargs):
            statics = {}
            for name, value in kwargs.items():
                if isinstance(value, (str, int, float, bool, list, tuple, dict)):
                    statics[name] = value
            param_path = os.path.join(kwargs[kwargs_key], "params.yaml")
            yaml.dump(statics, open(param_path, "w"))
            return func(**kwargs)

        return wrapper

    return decorator


@app.command()
def load_config(path: str):
    train(**yaml.load(open(path, "r"), Loader=yaml.Loader))


@app.command()
@save_params("root_log_dir")
def train(
    *,
    name: str,
    root_log_dir: str,
    model_conf_path: str,
    lr: float,
    num_workers: int,
    weight_decay: float,
    betas: Tuple[float, float],
    warmup_epochs: int,
    shift_epoch: int,
    epoch: int,
    total_epochs: int,
    samples_per_epoch: int,
    silent: bool,
    device: str,
    amp: bool,
    root: str,
    exts: List[str],
    recursive: bool = False,
    absolute_path: bool = False,
    shuffle: bool = False,
    size: Tuple[int, int] = (256, 256),
    batch_size: int = 32,
    transform_conf_path: Optional[str] = None,
    log_frequency: int = 100,
) -> None:
    log_dir, run_dir, checkpoint_dir, out_dir = make_logdirs(root_log_dir, name)

    shift_steps = shift_epoch * samples_per_epoch
    steps_per_epoch = math.ceil(samples_per_epoch / batch_size)
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch
    model: Module = Available.load(model_conf_path)
    transform = Available.load(transform_conf_path) if transform_conf_path else None
    dataloader = DataLoader2(
        datapipe(
            root, exts, recursive, absolute_path, shuffle, size, batch_size, transform
        ),
        reading_service=MultiProcessingReadingService(num_workers=num_workers),
    )
    opt = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        foreach=True,
    )
    sched = LambdaLR(
        opt,
        lr_lambda=partial(
            cosine_lr,
            total=total_steps,
            warmup=warmup_steps,
            shift=shift_steps,
        ),
    )
    writer = SummaryWriter(run_dir)
    model.to(device)
    scaler = GradScaler(enabled=amp)
    global_step = epoch * samples_per_epoch
    for epoch in range(epoch, total_epochs):
        for batch_idx, images in enumerate(
            pbar := tqdm(dataloader, disable=silent, ncols=120, mininterval=10)
        ):
            model.train()
            images = images.to(device)
            with torch.autocast(device, torch.float16, enabled=amp):
                pred = model(images)
                loss = F.l1_loss(pred, images)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            clip_grad_norm_(model.parameters(), 1.0, foreach=True)
            scaler.step(opt)
            scaler.update()
            sched.step()
            writer.add_scalar("train/loss", loss.item(), global_step, new_style=True)
            pbar.set_description(
                f"Epoch: {epoch} Step: {global_step} Loss: {loss.item():.4f}",
                refresh=False,
            )
            if global_step % log_frequency == 0:
                fig = plot_train_results(
                    images[0], pred[0], global_step, epoch, loss.item()
                )
                writer.add_figure("train/result", fig, global_step)
                save_checkpoint(model, checkpoint_dir)
            global_step += 1
