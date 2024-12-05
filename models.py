from abc import ABC, abstractmethod
from math import floor
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import torch
from einops import pack, rearrange, unpack
from torch import Generator, Tensor, nn

from components import AppendChannelsPlugin, MelSpectrogram
from diffusion import ARVDiffusion, ARVSampler, VDiffusion, VSampler
from utils import (
    closest_power_2,
    default,
    downsample,
    exists,
    groupby,
    randn_like,
    upsample,
)


class DiffusionModel(nn.Module):
    """
    DiffusionModel 类用于构建扩散模型。

    初始化参数:
    - net_t (Callable): 网络模型的构造函数，用于生成网络实例。
    - diffusion_t (Callable, 可选): 扩散模型的构造函数，默认为 VDiffusion。
    - sampler_t (Callable, 可选): 采样器的构造函数，默认为 VSampler。
    - loss_fn (Callable, 可选): 损失函数，默认为均方误差损失函数。
    - dim (int, 可选): 数据的维度，默认为1。
    - **kwargs: 其他传递给网络模型、扩散模型和采样器的关键字参数。
    """
    def __init__(
        self,
        net_t: Callable,
        diffusion_t: Callable = VDiffusion,
        sampler_t: Callable = VSampler,
        loss_fn: Callable = torch.nn.functional.mse_loss,
        dim: int = 1,
        **kwargs,
    ):
        super().__init__()
        # 使用 groupby 函数将 kwargs 按照前缀分组
        # 提取扩散模型的关键字参数
        diffusion_kwargs, kwargs = groupby("diffusion_", kwargs)
        # 提取采样器的关键字参数
        sampler_kwargs, kwargs = groupby("sampler_", kwargs)

        # 实例化网络模型
        # 使用 net_t 构造网络模型
        self.net = net_t(dim=dim, **kwargs)
        
        # 实例化扩散模型
        # 使用 diffusion_t 构造扩散模型
        self.diffusion = diffusion_t(net=self.net, loss_fn=loss_fn, **diffusion_kwargs)

        # 实例化采样器
        # 使用 sampler_t 构造采样器
        self.sampler = sampler_t(net=self.net, **sampler_kwargs)

    def forward(self, *args, **kwargs) -> Tensor:
        """
        前向传播方法，执行扩散过程。

        参数:
        - *args: 位置参数。
        - **kwargs: 关键字参数。

        返回:
        - Tensor: 扩散模型的输出。
        """
        # 调用扩散模型的前向传播方法
        return self.diffusion(*args, **kwargs)

    @torch.no_grad()
    def sample(self, *args, **kwargs) -> Tensor:
        """
        采样方法，执行采样过程。

        参数:
        - *args: 位置参数。
        - **kwargs: 关键字参数。

        返回:
        - Tensor: 采样器的输出。
        """
        # 调用采样器的方法进行采样
        return self.sampler(*args, **kwargs)


class EncoderBase(nn.Module, ABC):
    """Abstract class for DiffusionAE encoder"""
    """
    EncoderBase 是一个抽象基类，用于 DiffusionAE 编码器。

    抽象方法:
    - __init__: 初始化方法，子类必须实现。
    """

    @abstractmethod
    def __init__(self):
        """
        抽象初始化方法，子类必须实现。
        """
        super().__init__()
        self.out_channels = None # 输出通道数
        self.downsample_factor = None # 下采样因子


class AdapterBase(nn.Module, ABC):
    """Abstract class for DiffusionAE encoder"""
    """
    AdapterBase 是一个抽象基类，用于 DiffusionAE 适配器。

    抽象方法:
    - encode: 编码方法，子类必须实现。
    - decode: 解码方法，子类必须实现。
    """

    @abstractmethod
    def encode(self, x: Tensor) -> Tensor:
        """
        抽象编码方法，子类必须实现。

        参数:
        - x (Tensor): 输入张量。

        返回:
        - Tensor: 编码后的张量。
        """
        pass

    @abstractmethod
    def decode(self, x: Tensor) -> Tensor:
        """
        抽象解码方法，子类必须实现。

        参数:
        - x (Tensor): 输入张量。

        返回:
        - Tensor: 解码后的张量。
        """
        pass


class DiffusionAE(DiffusionModel):
    """Diffusion Auto Encoder"""
    """
    DiffusionAE 类实现了扩散自编码器（Diffusion Auto Encoder）。

    初始化参数:
    - in_channels (int): 输入通道数。
    - channels (Sequence[int]): 每个层的通道数序列。
    - encoder (EncoderBase): 编码器实例，必须是 EncoderBase 的子类。
    - inject_depth (int): 注入编码器输出的深度（层数）。
    - latent_factor (Optional[int], 可选): 潜在空间的缩放因子。如果未提供，则默认为编码器的下采样因子。
    - adapter (Optional[AdapterBase], 可选): 适配器实例，用于调整输入。如果未提供，则不使用适配器。
    - **kwargs: 其他传递给父类的关键字参数。
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        encoder: EncoderBase,
        inject_depth: int,
        latent_factor: Optional[int] = None,
        adapter: Optional[AdapterBase] = None,
        **kwargs,
    ):
        # 构建上下文通道序列，在 inject_depth 位置注入编码器的输出通道数
        context_channels = [0] * len(channels)
        context_channels[inject_depth] = encoder.out_channels
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            context_channels=context_channels,
            **kwargs,
        )
        # 存储输入通道数
        self.in_channels = in_channels
        # 存储编码器实例
        self.encoder = encoder
        # 存储注入深度
        self.inject_depth = inject_depth
        # Optional custom latent factor and adapter
        # 设置潜在空间的缩放因子，默认使用编码器的下采样因子
        self.latent_factor = default(latent_factor, self.encoder.downsample_factor)
        # 如果提供了适配器，则将其 requires_grad 设置为 False（冻结适配器参数）
        self.adapter = adapter.requires_grad_(False) if exists(adapter) else None

    def forward(  # type: ignore
        self, x: Tensor, with_info: bool = False, **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Any]]:
        """
        前向传播方法，执行编码和扩散过程。

        参数:
        - x (Tensor): 输入张量。
        - with_info (bool, 可选): 是否返回编码信息，默认为 False。
        - **kwargs: 其他关键字参数。

        返回:
        - Union[Tensor, Tuple[Tensor, Any]]: 如果 with_info 为 True，则返回损失和编码信息；否则，仅返回损失。
        """
        # Encode input to latent channels
        # 使用编码器将输入编码为潜在空间
        latent, info = self.encode(x, with_info=True)
        # 在指定深度注入潜在空间
        channels = [None] * self.inject_depth + [latent]
        # Adapt input to diffusion if adapter provided
        # 如果提供了适配器，则使用适配器编码输入
        x = self.adapter.encode(x) if exists(self.adapter) else x
        # Compute diffusion loss
        # 计算扩散损失
        loss = super().forward(x, channels=channels, **kwargs)
        # 返回损失和编码信息
        return (loss, info) if with_info else loss

    def encode(self, *args, **kwargs):
        """
        编码方法，使用编码器将输入编码为潜在空间。

        参数:
        - *args: 位置参数。
        - **kwargs: 关键字参数。

        返回:
        - Tensor: 编码后的潜在空间。
        """
        # 调用编码器进行编码
        return self.encoder(*args, **kwargs)

    @torch.no_grad()
    def decode(
        self, latent: Tensor, generator: Optional[Generator] = None, **kwargs
    ) -> Tensor:
        """
        解码方法，将潜在空间解码为输出。

        参数:
        - latent (Tensor): 潜在空间张量。
        - generator (Optional[Generator], 可选): 随机数生成器，默认为 None。
        - **kwargs: 其他关键字参数。

        返回:
        - Tensor: 解码后的输出。
        """
        # 获取批次大小
        b = latent.shape[0]
        # 根据潜在空间的缩放因子，计算噪声长度
        noise_length = closest_power_2(latent.shape[2] * self.latent_factor)
        # Compute noise by inferring shape from latent length
        # 生成随机噪声，形状与潜在空间匹配
        noise = torch.randn(
            (b, self.in_channels, noise_length),
            device=latent.device,
            dtype=latent.dtype,
            generator=generator,
        )
        # Compute context from latent
        # 构建上下文通道序列，在指定深度注入潜在空间
        channels = [None] * self.inject_depth + [latent]  # type: ignore
        # Decode by sampling while conditioning on latent channels
        # 解码过程，采样并以潜在空间为条件
        out = super().sample(noise, channels=channels, **kwargs)
        # Decode output with adapter if provided
        # 如果提供了适配器，则使用适配器解码输出
        return self.adapter.decode(out) if exists(self.adapter) else out


class DiffusionUpsampler(DiffusionModel):
    """
    DiffusionUpsampler 类实现了扩散上采样器，用于放大图像。

    初始化参数:
    - in_channels (int): 输入通道数。
    - upsample_factor (int): 上采样因子。
    - net_t (Callable): 网络模型的构造函数。
    - **kwargs: 其他传递给父类的关键字参数。
    """
    def __init__(
        self,
        in_channels: int,
        upsample_factor: int,
        net_t: Callable,
        **kwargs,
    ):
        # 存储上采样因子
        self.upsample_factor = upsample_factor
        # 使用 AppendChannelsPlugin 包装网络模型，添加输入通道数作为附加通道
        super().__init__(
            net_t=AppendChannelsPlugin(net_t, channels=in_channels),
            in_channels=in_channels,
            **kwargs,
        )

    def reupsample(self, x: Tensor) -> Tensor:
        """
        重新上采样方法，对输入张量进行下采样和上采样。

        参数:
        - x (Tensor): 输入张量。

        返回:
        - Tensor: 重新上采样后的张量。
        """
        # 克隆输入张量
        x = x.clone()
        # 进行下采样
        x = downsample(x, factor=self.upsample_factor)
        # 进行上采样
        x = upsample(x, factor=self.upsample_factor)
        # 返回重新上采样后的张量
        return x

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:  # type: ignore
        """
        前向传播方法，执行扩散上采样过程。

        参数:
        - x (Tensor): 输入张量。
        - *args: 其他位置参数。
        - **kwargs: 其他关键字参数。

        返回:
        - Tensor: 上采样后的输出。
        """
        # 重新上采样输入张量
        reupsampled = self.reupsample(x)
        return super().forward(x, *args, append_channels=reupsampled, **kwargs)

    @torch.no_grad()
    def sample(  # type: ignore
        self, downsampled: Tensor, generator: Optional[Generator] = None, **kwargs
    ) -> Tensor:
        """
        采样方法，执行扩散上采样采样过程。

        参数:
        - downsampled (Tensor): 下采样后的输入张量。
        - generator (Optional[Generator], 可选): 随机数生成器，默认为 None。
        - **kwargs: 其他关键字参数。

        返回:
        - Tensor: 采样后的输出。
        """
        # 上采样下采样后的输入
        reupsampled = upsample(downsampled, factor=self.upsample_factor)
        # 生成随机噪声
        noise = randn_like(reupsampled, generator=generator)
        return super().sample(noise, append_channels=reupsampled, **kwargs)


class DiffusionVocoder(DiffusionModel):
    """
    DiffusionVocoder 类实现了基于扩散模型的声码器，用于将频谱图转换为音频波形。

    初始化参数:
    - net_t (Callable): 网络模型的构造函数，用于生成网络实例。
    - mel_channels (int): 梅尔频谱图的通道数。
    - mel_n_fft (int): 梅尔频谱图的 FFT 窗口大小。
    - mel_hop_length (Optional[int], 可选): 梅尔频谱图的帧移长度。如果未提供，则默认为 floor(mel_n_fft) // 4。
    - mel_win_length (Optional[int], 可选): 梅尔频谱图的窗口长度。如果未提供，则默认为 mel_n_fft。
    - in_channels (int, 可选): 输入通道数，默认为1。此参数被忽略，因为通道数会自动批量处理。
    - **kwargs: 其他传递给父类的关键字参数。
    """
    def __init__(
        self,
        net_t: Callable,
        mel_channels: int,
        mel_n_fft: int,
        mel_hop_length: Optional[int] = None,
        mel_win_length: Optional[int] = None,
        in_channels: int = 1,  # Ignored: channels are automatically batched.
        **kwargs,
    ):
        # 设置默认帧移长度
        mel_hop_length = default(mel_hop_length, floor(mel_n_fft) // 4)
        # 设置默认窗口长度
        mel_win_length = default(mel_win_length, mel_n_fft)
        # 提取 MelSpectrogram 的关键字参数
        mel_kwargs, kwargs = groupby("mel_", kwargs)
        super().__init__(
            # 使用 AppendChannelsPlugin 包装网络模型，添加一个通道
            net_t=AppendChannelsPlugin(net_t, channels=1),
            in_channels=1,
            **kwargs,
        )
        # 初始化 MelSpectrogram，用于将音频波形转换为梅尔频谱图
        self.to_spectrogram = MelSpectrogram(
            n_fft=mel_n_fft, # FFT 窗口大小
            hop_length=mel_hop_length, # 帧移长度
            win_length=mel_win_length, # 窗口长度
            n_mel_channels=mel_channels, # 梅尔频谱图的通道数
            **mel_kwargs,
        )
        # 初始化转置卷积层，用于将梅尔频谱图转换为音频波形
        self.to_flat = nn.ConvTranspose1d(
            in_channels=mel_channels,
            out_channels=1,
            kernel_size=mel_win_length,
            stride=mel_hop_length,
            padding=(mel_win_length - mel_hop_length) // 2,
            bias=False,
        )

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:  # type: ignore
        """
        前向传播方法，执行频谱图到音频波形的转换。

        参数:
        - x (Tensor): 输入音频波形。
        - *args: 其他位置参数。
        - **kwargs: 其他关键字参数。

        返回:
        - Tensor: 转换后的音频波形。
        """
        # Get spectrogram, pack channels and flatten
        # 将音频波形转换为梅尔频谱图，重塑形状并展平通道
        spectrogram = rearrange(self.to_spectrogram(x), "b c f l -> (b c) f l")
        # 通过转置卷积层转换为音频波形
        spectrogram_flat = self.to_flat(spectrogram)
        # Pack wave channels
        # 重新调整音频波形的形状，添加通道维度
        x = rearrange(x, "b c t -> (b c) 1 t")
        return super().forward(x, *args, append_channels=spectrogram_flat, **kwargs)

    @torch.no_grad()
    def sample(  # type: ignore
        self, spectrogram: Tensor, generator: Optional[Generator] = None, **kwargs
    ) -> Tensor:  # type: ignore
        """
        采样方法，执行频谱图到音频波形的采样过程。

        参数:
        - spectrogram (Tensor): 输入频谱图。
        - generator (Optional[Generator], 可选): 随机数生成器，默认为 None。
        - **kwargs: 其他关键字参数。

        返回:
        - Tensor: 生成的音频波形。
        """
        # 打包通道并展平频谱图
        # Pack channels and flatten spectrogram
        spectrogram, ps = pack([spectrogram], "* f l")
        # 通过转置卷积层转换为音频波形
        spectrogram_flat = self.to_flat(spectrogram)
        # Get start noise and sample
        # 生成随机噪声并采样
        noise = randn_like(spectrogram_flat, generator=generator)
        waveform = super().sample(noise, append_channels=spectrogram_flat, **kwargs)
        # Unpack wave channels
        # 解包音频波形的通道
        waveform = rearrange(waveform, "... 1 t -> ... t")
        waveform = unpack(waveform, ps, "* t")[0]
        # 返回生成的音频波形
        return waveform


class DiffusionAR(DiffusionModel):
    """
    DiffusionAR 类实现了基于自回归的扩散模型，用于处理序列数据。

    初始化参数:
    - in_channels (int): 输入通道数。
    - length (int): 输入序列的总长度。
    - num_splits (int): 将序列分割成的部分数量。
    - diffusion_t (Callable, 可选): 扩散模型的构造函数，默认为 ARVDiffusion。
    - sampler_t (Callable, 可选): 采样器的构造函数，默认为 ARVSampler。
    - **kwargs: 其他传递给父类的关键字参数。
    """
    def __init__(
        self,
        in_channels: int,
        length: int,
        num_splits: int,
        diffusion_t: Callable = ARVDiffusion,
        sampler_t: Callable = ARVSampler,
        **kwargs,
    ):
        super().__init__(
            # 输入通道数增加1，用于时间条件
            in_channels=in_channels + 1,
            # 输出通道数
            out_channels=in_channels,
            # 扩散模型的构造函数
            diffusion_t=diffusion_t,
            # 扩散模型的长度
            diffusion_length=length,
            # 扩散模型的分割数量
            diffusion_num_splits=num_splits,
            # 采样器的构造函数
            sampler_t=sampler_t,
            # 采样器的输入通道数
            sampler_in_channels=in_channels,
            # 采样器的长度
            sampler_length=length,
            # 采样器的分割数量
            sampler_num_splits=num_splits,
            # 不使用时间条件
            use_time_conditioning=False,
            # 不使用调制
            use_modulation=False,
            **kwargs,
        )
