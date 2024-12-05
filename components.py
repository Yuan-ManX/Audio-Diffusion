from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F
from audio_blocks import (
    ClassifierFreeGuidancePlugin,
    Conv,
    Module,
    TextConditioningPlugin,
    TimeConditioningPlugin,
    default,
    exists,
)
from audio_unet import (
    AttentionItem,
    CrossAttentionItem,
    InjectChannelsItem,
    ModulationItem,
    ResnetItem,
    SkipCat,
    SkipModulate,
    XBlock,
    XUNet,
)
from einops import pack, unpack
from torch import Tensor, nn
from torchaudio import transforms



def UNetV0(
    dim: int,
    in_channels: int,
    channels: Sequence[int],
    factors: Sequence[int],
    items: Sequence[int],
    attentions: Optional[Sequence[int]] = None,
    cross_attentions: Optional[Sequence[int]] = None,
    context_channels: Optional[Sequence[int]] = None,
    attention_features: Optional[int] = None,
    attention_heads: Optional[int] = None,
    embedding_features: Optional[int] = None,
    resnet_groups: int = 8,
    use_modulation: bool = True,
    modulation_features: int = 1024,
    embedding_max_length: Optional[int] = None,
    use_time_conditioning: bool = True,
    use_embedding_cfg: bool = False,
    use_text_conditioning: bool = False,
    out_channels: Optional[int] = None,
):
    """
    UNetV0 函数用于构建一个 UNet 模型。

    参数:
    - dim (int): 数据的维度。
    - in_channels (int): 输入通道数。
    - channels (Sequence[int]): 每个层的通道数序列。
    - factors (Sequence[int]): 每个层的下采样因子序列。
    - items (Sequence[int]): 每个层的重复次数序列。
    - attentions (Optional[Sequence[int]], 可选): 每个层是否使用注意力机制。默认为 None。
    - cross_attentions (Optional[Sequence[int]], 可选): 每个层是否使用交叉注意力机制。默认为 None。
    - context_channels (Optional[Sequence[int]], 可选): 每个层的上下文通道数序列。默认为 None。
    - attention_features (Optional[int], 可选): 注意力特征维度。默认为 None。
    - attention_heads (Optional[int], 可选): 注意力头的数量。默认为 None.
    - embedding_features (Optional[int], 可选): 嵌入特征的维度。默认为 None。
    - resnet_groups (int, 可选): ResNet 组的数量，默认为8。
    - use_modulation (bool, 可选): 是否使用调制模块，默认为 True。
    - modulation_features (int, 可选): 调制特征的维度，默认为1024。
    - embedding_max_length (Optional[int], 可选): 嵌入的最大长度。
    - use_time_conditioning (bool, 可选): 是否使用时间条件，默认为 True。
    - use_embedding_cfg (bool, 可选): 是否使用嵌入配置控制，默认为 False。
    - use_text_conditioning (bool, 可选): 是否使用文本条件，默认为 False。
    - out_channels (Optional[int], 可选): 输出通道数。

    返回:
    - nn.Module: 构建好的 UNet 模型。
    """
    # Set defaults and check lengths
    num_layers = len(channels)
    # 如果未提供 attentions，则默认为全0列表
    attentions = default(attentions, [0] * num_layers)
    # 如果未提供 cross_attentions，则默认为全0列表
    cross_attentions = default(cross_attentions, [0] * num_layers)
    # 如果未提供 context_channels，则默认为全0列表
    context_channels = default(context_channels, [0] * num_layers)
    # 打包所有序列
    xs = (channels, factors, items, attentions, cross_attentions, context_channels)
    # 确保所有序列长度相同
    assert all(len(x) == num_layers for x in xs)  # type: ignore

    # Define UNet type
    # 定义 UNet 类型
    UNetV0 = XUNet

    if use_embedding_cfg:
        msg = "use_embedding_cfg requires embedding_max_length"
        # 确保提供了 embedding_max_length
        assert exists(embedding_max_length), msg
        # 应用 Classifier-Free Guidance 插件
        UNetV0 = ClassifierFreeGuidancePlugin(UNetV0, embedding_max_length)

    if use_text_conditioning:
        # 应用文本条件插件
        UNetV0 = TextConditioningPlugin(UNetV0)

    if use_time_conditioning:
        # 确保 use_modulation 为 True
        assert use_modulation, "use_time_conditioning requires use_modulation=True"
        # 应用时间条件插件
        UNetV0 = TimeConditioningPlugin(UNetV0)

    # Build
    # 构建 UNet 模型
    return UNetV0(
        dim=dim,
        in_channels=in_channels,
        out_channels=out_channels,
        blocks=[
            XBlock(
                channels=channels,
                factor=factor,
                context_channels=ctx_channels,
                items=(
                    [ResnetItem] # ResNet 块
                    + [ModulationItem] * use_modulation # 调制块（如果 use_modulation 为 True）
                    + [InjectChannelsItem] * (ctx_channels > 0) # 注入上下文通道块（如果 context_channels > 0）
                    + [AttentionItem] * att # 注意力块（根据 attentions 序列）
                    + [CrossAttentionItem] * cross # 交叉注意力块（根据 cross_attentions 序列）
                )
                * items, # 根据 items 序列重复模块
            )
            # 遍历所有序列
            for channels, factor, items, att, cross, ctx_channels in zip(*xs)  # type: ignore # noqa 
        ],
        # 选择跳跃连接类型
        skip_t=SkipModulate if use_modulation else SkipCat, 
        # 注意力特征维度
        attention_features=attention_features,
        # 注意力头的数量
        attention_heads=attention_heads,
        # 嵌入特征的维度
        embedding_features=embedding_features,
        # 调制特征的维度
        modulation_features=modulation_features,
        # ResNet 组的数量
        resnet_groups=resnet_groups,
    )



################################################ Plugins ################################################ 


def LTPlugin(
    net_t: Callable, num_filters: int, window_length: int, stride: int
) -> Callable[..., nn.Module]:
    """Learned Transform Plugin"""
    """
    Learned Transform Plugin（学习变换插件）。

    参数:
    - net_t (Callable): 原始网络类型。
    - num_filters (int): 滤波器数量。
    - window_length (int): 窗口长度。
    - stride (int): 步幅。

    返回:
    - Callable[..., nn.Module]: 一个函数，返回带有学习变换的模型。
    """

    def Net(
        dim: int, in_channels: int, out_channels: Optional[int] = None, **kwargs
    ) -> nn.Module:
        out_channels = default(out_channels, in_channels)
        in_channel_transform = in_channels * num_filters # 变换后的输入通道数
        out_channel_transform = out_channels * num_filters  # 变换后的输出通道数

        # 计算填充大小以保持输出尺寸        
        padding = window_length // 2 - stride // 2

        # 编码器：卷积层，将输入通道数转换为变换后的输入通道数
        encode = Conv(
            dim=dim,
            in_channels=in_channels,
            out_channels=in_channel_transform,
            kernel_size=window_length,
            stride=stride,
            padding=padding,
            padding_mode="reflect", # 使用反射填充
            bias=False, # 不使用偏置
        )

        # 解码器：转置卷积层，将变换后的输出通道数转换回原始输出通道数
        decode = nn.ConvTranspose1d(
            in_channels=out_channel_transform,
            out_channels=out_channels,  # type: ignore
            kernel_size=window_length,
            stride=stride,
            padding=padding,
            bias=False,
        )

        # 初始化原始网络，通道数已经变换
        net = net_t(  # type: ignore
            dim=dim,
            in_channels=in_channel_transform,
            out_channels=out_channel_transform,
            **kwargs
        )

        def forward(x: Tensor, *args, **kwargs):
            """
            前向传播方法。

            参数:
            - x (Tensor): 输入张量。
            - *args: 其他位置参数。
            - **kwargs: 其他关键字参数。

            返回:
            - Tensor: 输出张量。
            """
            x = encode(x)
            x = net(x, *args, **kwargs)
            x = decode(x)
            return x

        return Module([encode, decode, net], forward)

    return Net


def AppendChannelsPlugin(
    net_t: Callable, # 原始网络类型
    channels: int, # 要添加的通道数
):
    """
    AppendChannelsPlugin（添加通道插件）。

    参数:
    - net_t (Callable): 原始网络类型。
    - channels (int): 要添加的通道数。

    返回:
    - Callable[..., nn.Module]: 一个函数，返回带有添加通道的模型。
    """
    def Net(
        in_channels: int, out_channels: Optional[int] = None, **kwargs
    ) -> nn.Module:
        out_channels = default(out_channels, in_channels)
        # 初始化原始网络，输入通道数增加
        net = net_t(  # type: ignore
            in_channels=in_channels + channels, out_channels=out_channels, **kwargs
        )

        def forward(x: Tensor, *args, append_channels: Tensor, **kwargs):
            """
            前向传播方法。

            参数:
            - x (Tensor): 输入张量。
            - *args: 其他位置参数。
            - append_channels (Tensor): 要添加的通道张量。
            - **kwargs: 其他关键字参数。

            返回:
            - Tensor: 输出张量。
            """
            # 在通道维度上连接输入张量和添加的通道
            x = torch.cat([x, append_channels], dim=1)
            return net(x, *args, **kwargs)

        return Module([net], forward)

    return Net



################################################ Other ################################################ 


class MelSpectrogram(nn.Module):
    """
    MelSpectrogram 类用于将音频波形转换为梅尔频谱图。

    初始化参数:
    - n_fft (int): FFT 窗口大小。
    - hop_length (int): 帧移长度。
    - win_length (int): 窗口长度。
    - sample_rate (int): 采样率。
    - n_mel_channels (int): 梅尔频谱图的通道数。
    - center (bool, 可选): 是否居中对齐，默认为 False。
    - normalize (bool, 可选): 是否进行归一化，默认为 False。
    - normalize_log (bool, 可选): 是否进行对数归一化，默认为 False。
    """
    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        sample_rate: int,
        n_mel_channels: int,
        center: bool = False,
        normalize: bool = False,
        normalize_log: bool = False,
    ):
        super().__init__()
        # 计算填充大小
        self.padding = (n_fft - hop_length) // 2
        # 是否进行归一化
        self.normalize = normalize
        # 是否进行对数归一化
        self.normalize_log = normalize_log
        # 帧移长度
        self.hop_length = hop_length

        # 初始化 Spectrogram 模块，将音频波形转换为频谱图
        self.to_spectrogram = transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            center=center,
            power=None,
        )

        # 初始化 MelScale 模块，将频谱图转换为梅尔频谱图
        self.to_mel_scale = transforms.MelScale(
            n_mels=n_mel_channels, n_stft=n_fft // 2 + 1, sample_rate=sample_rate
        )

    def forward(self, waveform: Tensor) -> Tensor:
        """
        前向传播方法，将音频波形转换为梅尔频谱图。

        参数:
        - waveform (Tensor): 输入音频波形，形状为 (batch_size, channels, samples)。

        返回:
        - Tensor: 梅尔频谱图，形状为 (batch_size, n_mel_channels, time_steps)。
        """
        # Pack non-time dimension
        # 打包非时间维度
        waveform, ps = pack([waveform], "* t")
        # Pad waveform
        # 对音频波形进行填充
        waveform = F.pad(waveform, [self.padding] * 2, mode="reflect")
        # Compute STFT
        # 计算 STFT
        spectrogram = self.to_spectrogram(waveform)
        # Compute magnitude
        # 计算幅度谱
        spectrogram = torch.abs(spectrogram)
        # Convert to mel scale
        # 转换为梅尔频谱图
        mel_spectrogram = self.to_mel_scale(spectrogram)
        # Normalize
        # 进行归一化
        if self.normalize:
            # 归一化到 [0, 1]
            mel_spectrogram = mel_spectrogram / torch.max(mel_spectrogram)
            # 对数归一化
            mel_spectrogram = 2 * torch.pow(mel_spectrogram, 0.25) - 1
        if self.normalize_log:
            # 对数归一化
            mel_spectrogram = torch.log(torch.clamp(mel_spectrogram, min=1e-5))
        # Unpack non-spectrogram dimension
        # 解包非频谱维度
        # 返回梅尔频谱图
        return unpack(mel_spectrogram, ps, "* f l")[0]
    
