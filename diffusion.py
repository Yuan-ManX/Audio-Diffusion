from math import pi
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor
from tqdm import tqdm

from utils import default



################################################ Distributions ################################################ 


class Distribution:
    """Interface used by different distributions"""
    """
    Distribution 类是一个接口，用于定义不同分布的通用方法。

    调用方法:
    - __call__(num_samples: int, device: torch.device) -> Tensor:
      生成指定数量的样本并将其移动到指定的设备上。
      具体实现由子类完成。
    """

    def __call__(self, num_samples: int, device: torch.device):
        """
        生成指定数量的样本并将其移动到指定的设备上。

        参数:
        - num_samples (int): 需要生成的样本数量。
        - device (torch.device): 样本将被移动到的设备。

        返回:
        - Tensor: 生成的张量样本。

        抛出:
        - NotImplementedError: 如果子类没有实现该方法。
        """
        raise NotImplementedError()


class UniformDistribution(Distribution):
    """
    UniformDistribution 类实现了均匀分布。

    初始化参数:
    - vmin (float, 可选): 均匀分布的下界，默认为0.0。
    - vmax (float, 可选): 均匀分布的上界，默认为1.0。
    """
    def __init__(self, vmin: float = 0.0, vmax: float = 1.0):
        super().__init__() # 调用父类的初始化方法
        self.vmin, self.vmax = vmin, vmax # 存储下界和上界

    def __call__(self, num_samples: int, device: torch.device = torch.device("cpu")):
        """
        生成均匀分布的样本。

        参数:
        - num_samples (int): 需要生成的样本数量。
        - device (torch.device, 可选): 样本将被移动到的设备，默认为 CPU。

        返回:
        - Tensor: 均匀分布的样本，形状为 (num_samples,)。
        """
        # 获取上界和下界
        vmax, vmin = self.vmax, self.vmin
        # 生成均匀分布的随机数并缩放到 [vmin, vmax] 范围
        return (vmax - vmin) * torch.rand(num_samples, device=device) + vmin



################################################ Diffusion Methods ################################################ 


def pad_dims(x: Tensor, ndim: int) -> Tensor:
    # Pads additional ndims to the right of the tensor
    """
    在张量的右侧填充额外的维度。

    参数:
    - x (Tensor): 输入张量。
    - ndim (int): 需要填充的维度数量。

    返回:
    - Tensor: 填充后的张量，形状为 x.shape + (1,)*ndim。
    """
    # 在张量形状后面添加 ndim 个 1
    return x.view(*x.shape, *((1,) * ndim))


def clip(x: Tensor, dynamic_threshold: float = 0.0):
    """
    对张量进行裁剪处理。

    参数:
    - x (Tensor): 输入张量。
    - dynamic_threshold (float, 可选): 动态阈值，默认为0.0。
                                       如果为0.0，则使用静态阈值 [-1.0, 1.0]。

    返回:
    - Tensor: 裁剪后的张量。

    动态阈值裁剪步骤:
    - 计算每个批次元素的绝对值的动态阈值分位数。
    - 最小阈值为1.0。
    - 将张量裁剪到 [-scale, scale] 并除以 scale。
    """
    if dynamic_threshold == 0.0:
        # 如果动态阈值为0，则使用静态阈值 [-1.0, 1.0]
        return x.clamp(-1.0, 1.0)
    else:
        # Dynamic thresholding
        # Find dynamic threshold quantile for each batch
        # 动态阈值裁剪
        # 将 x 重塑为 (batch_size, ...)
        x_flat = rearrange(x, "b ... -> b (...)")
        # 计算每个批次元素绝对值的动态阈值分位数
        scale = torch.quantile(x_flat.abs(), dynamic_threshold, dim=-1)
        # Clamp to a min of 1.0
        # 最小阈值为1.0
        scale.clamp_(min=1.0)
        # Clamp all values and scale
        # 将 scale 张量填充到与 x 相同的维度
        scale = pad_dims(scale, ndim=x.ndim - scale.ndim)
        # 将 x 裁剪到 [-scale, scale] 并除以 scale
        x = x.clamp(-scale, scale) / scale
        # 返回裁剪后的张量
        return x


def extend_dim(x: Tensor, dim: int):
    # e.g. if dim = 4: shape [b] => [b, 1, 1, 1],
    """
    扩展张量的维度。

    例如，如果 dim = 4，形状 [b] => [b, 1, 1, 1]。

    参数:
    - x (Tensor): 输入张量。
    - dim (int): 目标维度。

    返回:
    - Tensor: 扩展后的张量，形状为 x.shape + (1,)*(dim - x.ndim)。
    """
    # 在张量形状后面添加 (dim - x.ndim) 个 1
    return x.view(*x.shape + (1,) * (dim - x.ndim))


class Diffusion(nn.Module):
    """Interface used by different diffusion methods"""
    """
    Diffusion 类是一个接口，用于定义不同扩散方法的通用方法。
    """

    pass


class VDiffusion(Diffusion):
    """
    VDiffusion 类实现了基于速度的扩散模型。

    初始化参数:
    - net (nn.Module): 神经网络模型，用于预测速度。
    - sigma_distribution (Distribution, 可选): 噪声分布，默认为均匀分布。
    - loss_fn (Any, 可选): 损失函数，默认为均方误差损失。
    """
    def __init__(
        self, net: nn.Module, sigma_distribution: Distribution = UniformDistribution(), loss_fn: Any = F.mse_loss
    ):
        super().__init__()
        # 神经网络模型
        self.net = net
        # 噪声分布
        self.sigma_distribution = sigma_distribution
        # 损失函数
        self.loss_fn = loss_fn

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        """
        根据 sigmas 计算 alpha 和 beta。

        参数:
        - sigmas (Tensor): 标准差张量。

        返回:
        - Tuple[Tensor, Tensor]: alpha 和 beta 的元组。
        """
        # 计算角度
        angle = sigmas * pi / 2
        # 计算 alpha 和 beta
        alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta

    def forward(self, x: Tensor, **kwargs) -> Tensor:  # type: ignore
        """
        前向传播方法，执行扩散过程。

        参数:
        - x (Tensor): 输入张量。
        - **kwargs: 其他关键字参数。

        返回:
        - Tensor: 损失值。
        """
        # 获取批次大小和设备
        batch_size, device = x.shape[0], x.device
        # Sample amount of noise to add for each batch element
        # 为每个批次元素采样添加的噪声量
        sigmas = self.sigma_distribution(num_samples=batch_size, device=device)
        # 将 sigmas 扩展到与 x 相同的维度
        sigmas_batch = extend_dim(sigmas, dim=x.ndim)

        # Get noise
        # 生成与 x 形状相同的噪声
        noise = torch.randn_like(x)
        # Combine input and noise weighted by half-circle
        # 使用半圆权重组合输入和噪声
        alphas, betas = self.get_alpha_beta(sigmas_batch)
        # 生成带噪声的输入
        x_noisy = alphas * x + betas * noise
        # 计算目标速度
        v_target = alphas * noise - betas * x
        # Predict velocity and return loss
        # 预测速度并返回损失
        v_pred = self.net(x_noisy, sigmas, **kwargs)
        # 计算损失并返回
        return self.loss_fn(v_pred, v_target)


class ARVDiffusion(Diffusion):
    """
    ARVDiffusion 类实现了基于自回归的速度扩散模型。

    初始化参数:
    - net (nn.Module): 神经网络模型，用于预测速度。
    - length (int): 输入序列的总长度。
    - num_splits (int): 将序列分割成的部分数量。
    - loss_fn (Any, 可选): 损失函数，默认为均方误差损失。
    """
    def __init__(self, net: nn.Module, length: int, num_splits: int, loss_fn: Any = F.mse_loss):
        super().__init__()
        # 确保长度可以被分割数量整除
        assert length % num_splits == 0, "length must be divisible by num_splits"
        self.net = net
        self.length = length
        self.num_splits = num_splits
        self.split_length = length // num_splits
        self.loss_fn = loss_fn

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        """
        根据 sigmas 计算 alpha 和 beta。

        参数:
        - sigmas (Tensor): 标准差张量。

        返回:
        - Tuple[Tensor, Tensor]: alpha 和 beta 的元组。
        """
        angle = sigmas * pi / 2
        alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """
        前向传播方法，执行自回归扩散过程。

        参数:
        - x (Tensor): 输入张量。
        - **kwargs: 其他关键字参数。

        返回:
        - Tensor: 损失值。
        """
        """Returns diffusion loss of v-objective with different noises per split"""
        # 获取批次大小、通道数、长度、设备和数据类型
        b, _, t, device, dtype = *x.shape, x.device, x.dtype
        # 确保输入长度与模型长度匹配
        assert t == self.length, "input length must match length"
        # Sample amount of noise to add for each split
        # 为每个分割部分采样添加的噪声量
        sigmas = torch.rand((b, 1, self.num_splits), device=device, dtype=dtype)
        # 为每个分割部分重复噪声
        sigmas = repeat(sigmas, "b 1 n -> b 1 (n l)", l=self.split_length)
        # Get noise
        # 生成与 x 形状相同的噪声
        noise = torch.randn_like(x)
        # Combine input and noise weighted by half-circle
        # 使用半圆权重组合输入和噪声
        alphas, betas = self.get_alpha_beta(sigmas)
        # 生成带噪声的输入
        x_noisy = alphas * x + betas * noise
        # 计算目标速度
        v_target = alphas * noise - betas * x
        # Sigmas will be provided as additional channel
        # 将 sigmas 作为额外的通道添加到 x_noisy 中
        channels = torch.cat([x_noisy, sigmas], dim=1)
        # Predict velocity and return loss
        # 预测速度并返回损失
        v_pred = self.net(channels, **kwargs)
        return self.loss_fn(v_pred, v_target)



################################################ Schedules ################################################ 


class Schedule(nn.Module):
    """Interface used by different sampling schedules"""
    """
    Schedule 类是一个接口，用于定义不同采样调度策略的通用方法。

    前向传播方法:
    - forward(num_steps: int, device: torch.device) -> Tensor:
      生成指定数量的调度步长，并将其移动到指定的设备上。
      具体实现由子类完成。
    """

    def forward(self, num_steps: int, device: torch.device) -> Tensor:
        """
        生成指定数量的调度步长，并将其移动到指定的设备上。

        参数:
        - num_steps (int): 需要生成的调度步长数量。
        - device (torch.device): 调度步长将被移动到的设备。

        返回:
        - Tensor: 生成的调度步长张量。

        抛出:
        - NotImplementedError: 如果子类没有实现该方法。
        """
        raise NotImplementedError()


class LinearSchedule(Schedule):
    """
    LinearSchedule 类实现了线性调度策略。

    初始化参数:
    - start (float, 可选): 调度的起始值，默认为1.0。
    - end (float, 可选): 调度的结束值，默认为0.0。
    """
    def __init__(self, start: float = 1.0, end: float = 0.0):
        super().__init__()
        # 存储起始值和结束值
        self.start, self.end = start, end

    def forward(self, num_steps: int, device: Any) -> Tensor:
        """
        生成线性调度的调度步长。

        参数:
        - num_steps (int): 需要生成的调度步长数量。
        - device (Any): 调度步长将被移动到的设备。

        返回:
        - Tensor: 线性调度的调度步长张量，形状为 (num_steps,)。
        """
        # 生成线性调度的调度步长
        return torch.linspace(self.start, self.end, num_steps, device=device)



################################################ Samplers ################################################ 


class Sampler(nn.Module):
    """
    Sampler 类是一个接口，用于定义不同采样器的通用方法。
    """
    pass


class VSampler(Sampler):
    """
    VSampler 类实现了基于速度的采样器，用于生成图像样本。

    初始化参数:
    - net (nn.Module): 用于预测速度的神经网络模型。
    - schedule (Schedule, 可选): 采样调度策略，默认为 LinearSchedule。
    """
    # 支持的扩散类型
    diffusion_types = [VDiffusion]

    def __init__(self, net: nn.Module, schedule: Schedule = LinearSchedule()):
        super().__init__()
        # 神经网络模型
        self.net = net
        # 采样调度策略
        self.schedule = schedule

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        """
        根据 sigmas 计算 alpha 和 beta。

        参数:
        - sigmas (Tensor): 标准差张量。

        返回:
        - Tuple[Tensor, Tensor]: alpha 和 beta 的元组。
        """
        angle = sigmas * pi / 2
        alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta

    @torch.no_grad()
    def forward(  # type: ignore
        self, x_noisy: Tensor, num_steps: int, show_progress: bool = False, **kwargs
    ) -> Tensor:
        """
        前向传播方法，执行采样过程。

        参数:
        - x_noisy (Tensor): 带噪声的输入图像。
        - num_steps (int): 采样步骤数量。
        - show_progress (bool, 可选): 是否显示进度条，默认为 False。
        - **kwargs: 其他关键字参数。

        返回:
        - Tensor: 生成的图像样本。
        """
        # 获取批次大小
        b = x_noisy.shape[0]
        # 生成调度步长并扩展到与 x_noisy 相同的批次大小
        sigmas = self.schedule(num_steps + 1, device=x_noisy.device)
        sigmas = repeat(sigmas, "i -> i b", b=b)
        sigmas_batch = extend_dim(sigmas, dim=x_noisy.ndim + 1)
        alphas, betas = self.get_alpha_beta(sigmas_batch)
        # 创建进度条
        progress_bar = tqdm(range(num_steps), disable=not show_progress)

        for i in progress_bar:
            # 预测速度
            v_pred = self.net(x_noisy, sigmas[i], **kwargs)
            # 计算预测的图像
            x_pred = alphas[i] * x_noisy - betas[i] * v_pred
            # 计算预测的噪声
            noise_pred = betas[i] * x_noisy + alphas[i] * v_pred
            # 更新带噪声的图像
            x_noisy = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred
            # 更新进度条描述
            progress_bar.set_description(f"Sampling (noise={sigmas[i+1,0]:.2f})")
        # 返回生成的图像
        return x_noisy


class ARVSampler(Sampler):
    """
    ARVSampler 类实现了基于自回归的速度扩散采样器，用于生成图像样本。

    初始化参数:
    - net (nn.Module): 用于预测速度的神经网络模型。
    - in_channels (int): 输入通道数。
    - length (int): 输入序列的总长度。
    - num_splits (int): 将序列分割成的部分数量。
    """
    def __init__(self, net: nn.Module, in_channels: int, length: int, num_splits: int):
        super().__init__()
        assert length % num_splits == 0, "length must be divisible by num_splits"
        self.length = length
        self.in_channels = in_channels
        self.num_splits = num_splits
        self.split_length = length // num_splits
        self.net = net

    @property
    def device(self):
        return next(self.net.parameters()).device

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        """
        根据 sigmas 计算 alpha 和 beta。

        参数:
        - sigmas (Tensor): 标准差张量。

        返回:
        - Tuple[Tensor, Tensor]: alpha 和 beta 的元组。
        """
        angle = sigmas * pi / 2
        alpha = torch.cos(angle)
        beta = torch.sin(angle)
        return alpha, beta

    def get_sigmas_ladder(self, num_items: int, num_steps_per_split: int) -> Tensor:
        """
        生成自回归阶梯噪声。

        参数:
        - num_items (int): 项目数量。
        - num_steps_per_split (int): 每个分割部分的步骤数量。

        返回:
        - Tensor: 自回归阶梯噪声张量。
        """
        b, n, l, i = num_items, self.num_splits, self.split_length, num_steps_per_split
        # 只生成一半的阶梯，剩余部分为0，以保留一些上下文
        n_half = n // 2  # Only half ladder, rest is zero, to leave some context
        # 生成线性阶梯噪声
        sigmas = torch.linspace(1, 0, i * n_half, device=self.device)
        # 扩展到每个分割部分
        sigmas = repeat(sigmas, "(n i) -> i b 1 (n l)", b=b, l=l, n=n_half)
        # 翻转以使最低噪声水平优先
        sigmas = torch.flip(sigmas, dims=[-1])  # Lowest noise level first
        # 添加索引 i+1
        sigmas = F.pad(sigmas, pad=[0, 0, 0, 0, 0, 0, 0, 1])  # Add index i+1
        # 在索引 i+1 处循环回
        sigmas[-1, :, :, l:] = sigmas[0, :, :, :-l]  # Loop back at index i+1
        # 返回拼接后的阶梯噪声
        return torch.cat([torch.zeros_like(sigmas), sigmas], dim=-1)

    def sample_loop(
        self, current: Tensor, sigmas: Tensor, show_progress: bool = False, **kwargs
    ) -> Tensor:
        """
        采样循环，执行自回归采样过程。

        参数:
        - current (Tensor): 当前图像。
        - sigmas (Tensor): 噪声标准差。
        - show_progress (bool, 可选): 是否显示进度条，默认为 False。
        - **kwargs: 其他关键字参数。

        返回:
        - Tensor: 更新后的图像。
        """
        # 计算总步骤数
        num_steps = sigmas.shape[0] - 1
        # 计算 alpha 和 beta
        alphas, betas = self.get_alpha_beta(sigmas)
        progress_bar = tqdm(range(num_steps), disable=not show_progress)

        for i in progress_bar:
            # 合并当前图像和噪声
            channels = torch.cat([current, sigmas[i]], dim=1)
            # 预测速度
            v_pred = self.net(channels, **kwargs)
            # 计算预测的图像
            x_pred = alphas[i] * current - betas[i] * v_pred
            # 计算预测的噪声
            noise_pred = betas[i] * current + alphas[i] * v_pred
            # 更新当前图像
            current = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred
            # 更新进度条描述
            progress_bar.set_description(f"Sampling (noise={sigmas[i+1,0,0,0]:.2f})")
        # 返回最终图像
        return current

    def sample_start(self, num_items: int, num_steps: int, **kwargs) -> Tensor:
        """
        生成初始采样。

        参数:
        - num_items (int): 项目数量。
        - num_steps (int): 采样步骤数量。
        - **kwargs: 其他关键字参数。

        返回:
        - Tensor: 初始采样的图像。
        """
        b, c, t = num_items, self.in_channels, self.length
        # Same sigma schedule over all chunks
        # 生成线性阶梯噪声
        sigmas = torch.linspace(1, 0, num_steps + 1, device=self.device)
        sigmas = repeat(sigmas, "i -> i b 1 t", b=b, t=t)
        # 生成噪声
        noise = torch.randn((b, c, t), device=self.device) * sigmas[0]
        # Sample start
        # 开始采样
        return self.sample_loop(current=noise, sigmas=sigmas, **kwargs)

    @torch.no_grad()
    def forward(
        self,
        num_items: int,
        num_chunks: int,
        num_steps: int,
        start: Optional[Tensor] = None,
        show_progress: bool = False,
        **kwargs,
    ) -> Tensor:
        """
        前向传播方法，执行采样过程。

        参数:
        - num_items (int): 项目数量。
        - num_chunks (int): 分割块数量。
        - num_steps (int): 采样步骤数量。
        - start (Optional[Tensor], 可选): 初始图像，默认为 None。
        - show_progress (bool, 可选): 是否显示进度条，默认为 False。
        - **kwargs: 其他关键字参数。

        返回:
        - Tensor: 生成的图像。
        """
        assert_message = f"required at least {self.num_splits} chunks"
        assert num_chunks >= self.num_splits, assert_message

        # Sample initial chunks
        # 采样初始块
        start = self.sample_start(num_items=num_items, num_steps=num_steps, **kwargs)
        # Return start if only num_splits chunks
        # 如果只有 num_splits 个块，则返回初始块
        if num_chunks == self.num_splits:
            return start

        # Get sigmas for autoregressive ladder
        # 获取自回归阶梯噪声
        b, n = num_items, self.num_splits
        assert num_steps >= n, "num_steps must be greater than num_splits"
        sigmas = self.get_sigmas_ladder(
            num_items=b,
            num_steps_per_split=num_steps // self.num_splits,
        )
        alphas, betas = self.get_alpha_beta(sigmas)

        # Noise start to match ladder and set starting chunks
        # 匹配阶梯噪声并设置起始块
        start_noise = alphas[0] * start + betas[0] * torch.randn_like(start)
        chunks = list(start_noise.chunk(chunks=n, dim=-1))

        # Loop over ladder shifts
        # 循环进行阶梯偏移
        num_shifts = num_chunks  # - self.num_splits
        progress_bar = tqdm(range(num_shifts), disable=not show_progress)

        for j in progress_bar:
            # Decrease ladder noise of last n chunks
            # 减少最后 n 个块的阶梯噪声
            updated = self.sample_loop(
                current=torch.cat(chunks[-n:], dim=-1), sigmas=sigmas, **kwargs
            )
            # Update chunks
            # 更新块
            chunks[-n:] = list(updated.chunk(chunks=n, dim=-1))
            # Add fresh noise chunk
            # 添加新的噪声块
            shape = (b, self.in_channels, self.split_length)
            chunks += [torch.randn(shape, device=self.device)]

        return torch.cat(chunks[:num_chunks], dim=-1)


################################################ Inpainters ################################################ 


class Inpainter(nn.Module):
    """
    Inpainter 类是一个接口，用于定义不同图像修复方法的通用方法。
    """
    pass


class VInpainter(Inpainter):
    """
    VInpainter 类实现了基于速度的图像修复器，用于修复图像中的缺失区域。

    初始化参数:
    - net (nn.Module): 用于预测速度的神经网络模型。
    - schedule (Schedule, 可选): 采样调度策略，默认为 LinearSchedule。
    """
    # 支持的扩散类型
    diffusion_types = [VDiffusion]

    def __init__(self, net: nn.Module, schedule: Schedule = LinearSchedule()):
        """
        初始化 VInpainter 实例。

        参数:
        - net (nn.Module): 用于预测速度的神经网络模型。
        - schedule (Schedule, 可选): 采样调度策略，默认为 LinearSchedule。
        """
        super().__init__()
        self.net = net
        self.schedule = schedule

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        """
        根据 sigmas 计算 alpha 和 beta。

        参数:
        - sigmas (Tensor): 标准差张量。

        返回:
        - Tuple[Tensor, Tensor]: alpha 和 beta 的元组。
        """
        angle = sigmas * pi / 2
        alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta

    @torch.no_grad()
    def forward(  # type: ignore
        self,
        source: Tensor,
        mask: Tensor,
        num_steps: int,
        num_resamples: int,
        show_progress: bool = False,
        x_noisy: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        前向传播方法，执行图像修复过程。

        参数:
        - source (Tensor): 原始图像。
        - mask (Tensor): 掩码，指示需要修复的区域。
        - num_steps (int): 采样步骤数量。
        - num_resamples (int): 每个步骤中重采样的次数。
        - show_progress (bool, 可选): 是否显示进度条，默认为 False。
        - x_noisy (Optional[Tensor], 可选): 初始带噪声图像。如果未提供，则使用随机噪声初始化。
        - **kwargs: 其他关键字参数。

        返回:
        - Tensor: 修复后的图像。
        """
        # 如果未提供初始带噪声图像，则使用随机噪声初始化
        x_noisy = default(x_noisy, lambda: torch.randn_like(source))
        # 获取批次大小
        b = x_noisy.shape[0]
        # 生成调度步长并扩展到与 x_noisy 相同的批次大小
        sigmas = self.schedule(num_steps + 1, device=x_noisy.device)
        sigmas = repeat(sigmas, "i -> i b", b=b)
        sigmas_batch = extend_dim(sigmas, dim=x_noisy.ndim + 1)
        alphas, betas = self.get_alpha_beta(sigmas_batch)
        progress_bar = tqdm(range(num_steps), disable=not show_progress)

        for i in progress_bar:
            for r in range(num_resamples):
                # 预测速度
                v_pred = self.net(x_noisy, sigmas[i], **kwargs)
                # 计算预测的图像
                x_pred = alphas[i] * x_noisy - betas[i] * v_pred
                # 计算预测的噪声
                noise_pred = betas[i] * x_noisy + alphas[i] * v_pred
                # Renoise to current noise level if resampling
                # 如果是最后一次重采样，则更新噪声水平
                j = r == num_resamples - 1
                x_noisy = alphas[i + j] * x_pred + betas[i + j] * noise_pred
                s_noisy = alphas[i + j] * source + betas[i + j] * torch.randn_like(
                    source
                )
                # 生成带噪声的源图像
                # 应用掩码，保留修复区域
                x_noisy = s_noisy * mask + x_noisy * ~mask

            progress_bar.set_description(f"Inpainting (noise={sigmas[i+1,0]:.2f})")

        # 返回修复后的图像
        return x_noisy
