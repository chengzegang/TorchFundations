# SPDX-FileCopyrightText: 2023-present Zegang Cheng <zc2309@nyu.edu>
#
# SPDX-License-Identifier: MIT
import os
import torchvision  # type: ignore
import warnings
import torch
import torch.backends.cuda as cuda
import torch.backends.opt_einsum as opt_einsum
import torch.backends.cudnn as cudnn

warnings.filterwarnings("ignore")
torchvision.disable_beta_transforms_warning()
cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True
cudnn.benchmark = True
cuda.matmul.allow_fp16_reduced_precision_reduction = True
cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.set_default_dtype(torch.float32)
# torch.set_num_threads(1) uncomment this if your HPC does not allow multithreading
opt_einsum.enabled = True
torch.backends.cudnn.enabled = True
