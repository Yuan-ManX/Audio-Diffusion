from typing import Callable, List, Optional, Sequence, no_type_check

import torch
from torch import Tensor, nn

from audio_blocks import (
    Attention,
    Conv,
    ConvBlock,
    ConvNextV2Block,
    CrossAttention,
    Downsample,
    FeedForward,
    LinearAttentionBase,
    MergeAdd,
    MergeCat,
    MergeModulate,
    Modulation,
    Module,
    Packed,
    ResnetBlock,
    Select,
    Sequential,
    T,
    Upsample,
    UpsampleInterpolate,
    default,
    exists,
)



################################################ Items ################################################ 

# Selections for item forward parameters
# 定义 SelectX，用于选择性地传递参数 x
SelectX = Select(lambda x, *_: (x,))
# 定义 SelectXF，用于选择性地传递参数 x 和 f
SelectXF = Select(lambda x, f, *_: (x, f))
# 定义 SelectXE，用于选择性地传递参数 x 和 e
SelectXE = Select(lambda x, f, e, *_: (x, e))
# 定义 SelectXC，用于选择性地传递参数 x 和 c
SelectXC = Select(lambda x, f, e, c, *_: (x, c))



################################################ Downsample / Upsample ################################################ 


def DownsampleItem(
    dim: Optional[int] = None, # 数据的维度
    factor: Optional[int] = None, # 下采样因子
    in_channels: Optional[int] = None, # 输入通道数
    channels: Optional[int] = None, # 输出通道数
    downsample_width: int = 1, # 下采样卷积核宽度
    **kwargs, # 其他关键字参数
) -> nn.Module:
    """
    DownsampleItem 函数用于创建一个下采样模块。

    参数:
    - dim (Optional[int], 可选): 数据的维度。
    - factor (Optional[int], 可选): 下采样因子。
    - in_channels (Optional[int], 可选): 输入通道数。
    - channels (Optional[int], 可选): 输出通道数。
    - downsample_width (int, 可选): 下采样卷积核宽度，默认为1。
    - **kwargs: 其他传递给 Downsample 的关键字参数。

    返回:
    - nn.Module: 下采样模块。

    断言:
    - 必须提供 dim, factor, in_channels, channels 参数，否则抛出错误。
    """
    msg = "DownsampleItem requires dim, factor, in_channels, channels"
    assert (
        exists(dim) and exists(factor) and exists(in_channels) and exists(channels) # 确保所有必要参数存在
    ), msg

     # 使用 SelectX 选择性地传递参数给 Downsample
    Item = SelectX(Downsample)
    # 返回下采样模块
    return Item(  # type: ignore
        dim=dim,
        factor=factor,
        width=downsample_width,
        in_channels=in_channels,
        out_channels=channels,
    )


def UpsampleItem(
    dim: Optional[int] = None, # 数据的维度
    factor: Optional[int] = None, # 上采样因子
    channels: Optional[int] = None, # 输入通道数
    out_channels: Optional[int] = None, # 输出通道数
    upsample_mode: str = "nearest", # 上采样模式，默认为 "nearest"
    # 上采样卷积核大小，仅当 upsample_mode 不为 "transpose" 时使用
    upsample_kernel_size: int = 3,  # Used with upsample_mode != "transpose"
    # 上采样卷积核宽度，仅当 upsample_mode 为 "transpose" 时使用
    upsample_width: int = 1,  # Used with upsample_mode == "transpose"
    **kwargs, # 其他关键字参数
) -> nn.Module:
    """
    UpsampleItem 函数用于创建一个上采样模块。

    参数:
    - dim (Optional[int], 可选): 数据的维度。
    - factor (Optional[int], 可选): 上采样因子。
    - channels (Optional[int], 可选): 输入通道数。
    - out_channels (Optional[int], 可选): 输出通道数。
    - upsample_mode (str, 可选): 上采样模式，默认为 "nearest"。如果为 "transpose"，则使用转置卷积进行上采样。
    - upsample_kernel_size (int, 可选): 上采样卷积核大小，仅当 upsample_mode 不为 "transpose" 时使用，默认为3。
    - upsample_width (int, 可选): 上采样卷积核宽度，仅当 upsample_mode 为 "transpose" 时使用，默认为1。
    - **kwargs: 其他传递给 Upsample 或 UpsampleInterpolate 的关键字参数。

    返回:
    - nn.Module: 上采样模块。

    断言:
    - 必须提供 dim, factor, channels, out_channels 参数，否则抛出错误。
    """
    msg = "UpsampleItem requires dim, factor, channels, out_channels"
    assert (
        exists(dim) and exists(factor) and exists(channels) and exists(out_channels)
    ), msg # 确保所有必要参数存在
    if upsample_mode == "transpose":
        # 如果上采样模式为 "transpose"，则使用 Upsample
        Item = SelectX(Upsample)
        # 返回上采样模块
        return Item(  # type: ignore
            dim=dim,
            factor=factor,
            width=upsample_width,
            in_channels=channels,
            out_channels=out_channels,
        )
    else:
        # 否则，使用 UpsampleInterpolate
        Item = SelectX(UpsampleInterpolate)
        # 返回上采样模块
        return Item(  # type: ignore
            dim=dim,
            factor=factor,
            mode=upsample_mode,
            kernel_size=upsample_kernel_size,
            in_channels=channels,
            out_channels=out_channels,
        )



################################################ Main ################################################


def ResnetItem(
    dim: Optional[int] = None, # 数据的维度
    channels: Optional[int] = None, # 通道数
    resnet_groups: Optional[int] = None,  # ResNet 组的数量
    resnet_kernel_size: int = 3, # ResNet 卷积核大小，默认为3
    **kwargs,
) -> nn.Module:
    """
    ResnetItem 函数用于创建一个 ResNet 块模块。

    参数:
    - dim (Optional[int], 可选): 数据的维度。
    - channels (Optional[int], 可选): 通道数。
    - resnet_groups (Optional[int], 可选): ResNet 组的数量。
    - resnet_kernel_size (int, 可选): ResNet 卷积核大小，默认为3。
    - **kwargs: 其他传递给 ResnetBlock 的关键字参数。

    返回:
    - nn.Module: ResNet 块模块。

    断言:
    - 必须提供 dim, channels, resnet_groups 参数，否则抛出错误。
    """
    msg = "ResnetItem requires dim, channels, and resnet_groups"
    assert exists(dim) and exists(channels) and exists(resnet_groups), msg
    # 使用 SelectX 选择性地传递参数给 ResnetBlock
    Item = SelectX(ResnetBlock)
    # 初始化 ConvBlock，指定归一化层为 GroupNorm，组数为 resnet_groups
    conv_block_t = T(ConvBlock)(norm_t=T(nn.GroupNorm)(num_groups=resnet_groups))
    return Item(  # type: ignore
        dim=dim,
        in_channels=channels,
        out_channels=channels,
        kernel_size=resnet_kernel_size,
        conv_block_t=conv_block_t,
    )


def ConvNextV2Item(
    dim: Optional[int] = None,
    channels: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    """
    ConvNextV2Item 函数用于创建一个 ConvNextV2 块模块。

    参数:
    - dim (Optional[int], 可选): 数据的维度。
    - channels (Optional[int], 可选): 通道数。
    - **kwargs: 其他传递给 ConvNextV2Block 的关键字参数。

    返回:
    - nn.Module: ConvNextV2 块模块。

    断言:
    - 必须提供 dim 和 channels 参数，否则抛出错误。
    """
    msg = "ResnetItem requires dim and channels"
    assert exists(dim) and exists(channels), msg
    # 使用 SelectX 选择性地传递参数给 ConvNextV2Block
    Item = SelectX(ConvNextV2Block)
    # 返回 ConvNextV2 块模块
    return Item(dim=dim, channels=channels)  # type: ignore


def AttentionItem(
    channels: Optional[int] = None,
    attention_features: Optional[int] = None,
    attention_heads: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    """
    AttentionItem 函数用于创建一个注意力机制模块。

    参数:
    - channels (Optional[int], 可选): 通道数。
    - attention_features (Optional[int], 可选): 注意力特征维度。
    - attention_heads (Optional[int], 可选): 注意力头的数量。
    - **kwargs: 其他传递给 Attention 的关键字参数。

    返回:
    - nn.Module: 注意力机制模块。

    断言:
    - 必须提供 channels, attention_features, attention_heads 参数，否则抛出错误。
    """
    msg = "AttentionItem requires channels, attention_features, attention_heads"
    assert (
        exists(channels) and exists(attention_features) and exists(attention_heads)
    ), msg
    # 使用 SelectX 选择性地传递参数给 Attention
    Item = SelectX(Attention)
    # 返回封装后的注意力机制模块
    return Packed(
        Item(  # type: ignore
            features=channels,
            head_features=attention_features,
            num_heads=attention_heads,
        )
    )


def CrossAttentionItem(
    channels: Optional[int] = None, # 通道数
    attention_features: Optional[int] = None, # 注意力特征维度
    attention_heads: Optional[int] = None, # 注意力头的数量
    embedding_features: Optional[int] = None, # 嵌入特征维度
    **kwargs,
) -> nn.Module:
    """
    CrossAttentionItem 函数用于创建一个交叉注意力机制模块。

    参数:
    - channels (Optional[int], 可选): 通道数。
    - embedding_features (Optional[int], 可选): 嵌入特征维度。
    - attention_features (Optional[int], 可选): 注意力特征维度。
    - attention_heads (Optional[int], 可选): 注意力头的数量。
    - **kwargs: 其他传递给 CrossAttention 的关键字参数。

    返回:
    - nn.Module: 交叉注意力机制模块。

    断言:
    - 必须提供 channels, embedding_features, attention_features, attention_heads 参数，否则抛出错误。
    """
    msg = "CrossAttentionItem requires channels, embedding_features, attention_*"
    assert (
        exists(channels)
        and exists(embedding_features)
        and exists(attention_features)
        and exists(attention_heads)
    ), msg
    # 使用 SelectXE 选择性地传递参数给 CrossAttention
    Item = SelectXE(CrossAttention)
    # 返回封装后的交叉注意力机制模块
    return Packed(
        Item(  # type: ignore
            features=channels,
            head_features=attention_features,
            num_heads=attention_heads,
            context_features=embedding_features,
        )
    )


def ModulationItem(
    channels: Optional[int] = None, modulation_features: Optional[int] = None, **kwargs
) -> nn.Module:
    """
    ModulationItem 函数用于创建一个调制模块。

    参数:
    - channels (Optional[int], 可选): 通道数。
    - modulation_features (Optional[int], 可选): 调制特征维度。
    - **kwargs: 其他传递给 Modulation 的关键字参数。

    返回:
    - nn.Module: 调制模块。

    断言:
    - 必须提供 channels 和 modulation_features 参数，否则抛出错误。
    """
    msg = "ModulationItem requires channels, modulation_features"
    assert exists(channels) and exists(modulation_features), msg
    # 使用 SelectXF 选择性地传递参数给 Modulation
    Item = SelectXF(Modulation)
    # 返回封装后的调制模块
    return Packed(
        Item(in_features=channels, num_features=modulation_features)  # type: ignore
    )


def LinearAttentionItem(
    channels: Optional[int] = None,
    attention_features: Optional[int] = None,
    attention_heads: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    """
    LinearAttentionItem 函数用于创建一个线性注意力机制模块。

    参数:
    - channels (Optional[int], 可选): 通道数。
    - attention_features (Optional[int], 可选): 注意力特征维度。
    - attention_heads (Optional[int], 可选): 注意力头的数量。
    - **kwargs: 其他传递给 LinearAttentionBase 的关键字参数。

    返回:
    - nn.Module: 线性注意力机制模块。

    断言:
    - 必须提供 channels, attention_features, attention_heads 参数，否则抛出错误。
    """
    msg = "LinearAttentionItem requires attention_features and attention_heads"
    assert (
        exists(channels) and exists(attention_features) and exists(attention_heads)
    ), msg
    # 使用 SelectX 选择性地传递参数给线性注意力机制
    Item = SelectX(T(Attention)(attention_base_t=LinearAttentionBase))
    # 返回封装后的线性注意力机制模块
    return Packed(
        Item(  # type: ignore
            features=channels,
            head_features=attention_features,
            num_heads=attention_heads,
        )
    )


def LinearCrossAttentionItem(
    channels: Optional[int] = None,
    attention_features: Optional[int] = None,
    attention_heads: Optional[int] = None,
    embedding_features: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    """
    LinearCrossAttentionItem 函数用于创建一个线性交叉注意力机制模块。

    参数:
    - channels (Optional[int], 可选): 输入通道数。
    - attention_features (Optional[int], 可选): 注意力特征维度。
    - attention_heads (Optional[int], 可选): 注意力头的数量。
    - embedding_features (Optional[int], 可选): 嵌入特征维度。
    - **kwargs: 其他传递给 CrossAttention 的关键字参数。

    返回:
    - nn.Module: 线性交叉注意力机制模块。

    断言:
    - 必须提供 channels, embedding_features, attention_features, attention_heads 参数，否则抛出错误。
    """
    msg = "LinearCrossAttentionItem requires channels, embedding_features, attention_*"
    assert (
        exists(channels)
        and exists(embedding_features)
        and exists(attention_features)
        and exists(attention_heads)
    ), msg
    # 使用 SelectXE 选择性地传递参数给线性交叉注意力机制
    Item = SelectXE(T(CrossAttention)(attention_base_t=LinearAttentionBase))
    # 返回封装后的线性交叉注意力机制模块
    return Packed(
        Item(  # type: ignore
            features=channels,
            head_features=attention_features,
            num_heads=attention_heads,
            context_features=embedding_features,
        )
    )


def FeedForwardItem(
    channels: Optional[int] = None, attention_multiplier: Optional[int] = None, **kwargs
) -> nn.Module:
    """
    FeedForwardItem 函数用于创建一个前馈网络模块。

    参数:
    - channels (Optional[int], 可选): 输入通道数。
    - attention_multiplier (Optional[int], 可选): 注意力倍数，用于计算中间层维度。
    - **kwargs: 其他传递给 FeedForward 的关键字参数。

    返回:
    - nn.Module: 前馈网络模块。

    断言:
    - 必须提供 channels 和 attention_multiplier 参数，否则抛出错误。
    """
    msg = "FeedForwardItem requires channels, attention_multiplier"
    assert exists(channels) and exists(attention_multiplier), msg
    # 使用 SelectX 选择性地传递参数给前馈网络
    Item = SelectX(FeedForward)
    # 返回封装后的前馈网络模块
    return Packed(
        Item(features=channels, multiplier=attention_multiplier)  # type: ignore
    )


def InjectChannelsItem(
    dim: Optional[int] = None,
    channels: Optional[int] = None,
    depth: Optional[int] = None,
    context_channels: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    """
    InjectChannelsItem 函数用于注入上下文通道到主路径中。

    参数:
    - dim (Optional[int], 可选): 数据的维度。
    - channels (Optional[int], 可选): 输入通道数。
    - depth (Optional[int], 可选): 当前深度（层数）。
    - context_channels (Optional[int], 可选): 上下文通道数。
    - **kwargs: 其他传递给卷积层的关键字参数。

    返回:
    - nn.Module: 注入上下文通道的模块。

    断言:
    - 必须提供 dim, depth, channels, context_channels 参数，否则抛出错误。
    - context_channels 必须大于0。
    """
    msg = "InjectChannelsItem requires dim, depth, channels, context_channels"
    assert (
        exists(dim) and exists(depth) and exists(channels) and exists(context_channels)
    ), msg
    msg = "InjectChannelsItem requires context_channels > 0"
    assert context_channels > 0, msg

    # 初始化卷积层
    conv = Conv(
        dim=dim,
        in_channels=channels + context_channels, # 输入通道数为输入通道数加上下文通道数
        out_channels=channels, # 输出通道数与输入通道数相同
        kernel_size=1,  # 卷积核大小为1
    )

    @no_type_check
    def forward(x: Tensor, channels: Sequence[Optional[Tensor]]) -> Tensor:
        """
        前向传播方法，注入上下文通道到主路径。

        参数:
        - x (Tensor): 输入张量。
        - channels (Sequence[Optional[Tensor]]): 上下文通道序列。

        返回:
        - Tensor: 注入上下文通道后的输出张量。
        """
        msg_ = f"context `channels` at depth {depth} in forward"
        assert depth < len(channels), f"Required {msg_}" # 确保当前深度小于上下文通道序列长度
        # 获取当前深度的上下文通道
        context = channels[depth]
        # 计算期望的上下文通道形状
        shape = torch.Size([x.shape[0], context_channels, *x.shape[2:]])
        msg = f"Required {msg_} to be tensor of shape {shape}, found {context.shape}"
        assert torch.is_tensor(context) and context.shape == shape, msg
        # 将主路径输入和上下文通道连接起来，并通过卷积层处理
        return conv(torch.cat([x, context], dim=1)) + x

    # 返回封装后的注入上下文通道模块
    return SelectXC(Module)([conv], forward)  # type: ignore



################################################ Skip Adapters ################################################ 


def SkipAdapter(
    dim: Optional[int] = None,
    in_channels: Optional[int] = None,
    out_channels: Optional[int] = None,
    **kwargs,
):
    """
    SkipAdapter 函数用于创建一个跳跃连接适配器模块。

    参数:
    - dim (Optional[int], 可选): 数据的维度。
    - in_channels (Optional[int], 可选): 输入通道数。
    - out_channels (Optional[int], 可选): 输出通道数。
    - **kwargs: 其他传递给 Conv 或 nn.Identity 的关键字参数。

    返回:
    - nn.Module: 跳跃连接适配器模块。

    断言:
    - 必须提供 dim, in_channels, out_channels 参数，否则抛出错误。
    """
    msg = "SkipAdapter requires dim, in_channels, out_channels"
    assert exists(dim) and exists(in_channels) and exists(out_channels), msg

    # 使用 SelectX 选择性地传递参数给 Conv
    Item = SelectX(Conv)
    # 返回卷积层
    return (
        Item(  # type: ignore
            dim=dim,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
        )
        # 如果输入通道数不等于输出通道数，则使用 1x1 卷积层进行通道数匹配
        if in_channels != out_channels
        # 否则，使用恒等映射（Identity）作为跳跃连接
        else SelectX(nn.Identity)()
    )



################################################ Skip Connections ################################################ 


def SkipAdd(**kwargs) -> nn.Module:
    """
    SkipAdd 函数用于创建一个简单的跳跃连接模块，执行元素级相加。

    参数:
    - **kwargs: 其他传递给 MergeAdd 的关键字参数。

    返回:
    - nn.Module: 跳跃连接模块，执行元素级相加。
    """
    return MergeAdd()


def SkipCat(
    dim: Optional[int] = None,
    out_channels: Optional[int] = None,
    skip_scale: float = 2**-0.5,
    **kwargs,
) -> nn.Module:
    """
    SkipCat 函数用于创建一个跳跃连接模块，执行连接操作。

    参数:
    - dim (Optional[int], 可选): 数据的维度。
    - out_channels (Optional[int], 可选): 输出通道数。
    - skip_scale (float, 可选): 跳跃连接缩放因子，默认为 2**-0.5。
    - **kwargs: 其他传递给 MergeCat 的关键字参数。

    返回:
    - nn.Module: 跳跃连接模块，执行连接操作。

    断言:
    - 必须提供 dim 和 out_channels 参数，否则抛出错误。
    """
    msg = "SkipCat requires dim, out_channels"
    assert exists(dim) and exists(out_channels), msg
    return MergeCat(dim=dim, channels=out_channels, scale=skip_scale)


def SkipModulate(
    dim: Optional[int] = None,
    out_channels: Optional[int] = None,
    modulation_features: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    """
    SkipModulate 函数用于创建一个跳跃连接调制模块。

    参数:
    - dim (Optional[int], 可选): 数据的维度。
    - out_channels (Optional[int], 可选): 输出通道数。
    - modulation_features (Optional[int], 可选): 调制特征维度。
    - **kwargs: 其他传递给 MergeModulate 的关键字参数。

    返回:
    - nn.Module: 跳跃连接调制模块。

    断言:
    - 必须提供 dim, out_channels, modulation_features 参数，否则抛出错误。
    """
    msg = "SkipModulate requires dim, out_channels, modulation_features"
    assert exists(dim) and exists(out_channels) and exists(modulation_features), msg
    # 返回 MergeModulate 模块
    return MergeModulate(
        dim=dim, channels=out_channels, modulation_features=modulation_features
    )



################################################ Block ################################################ 


class Block(nn.Module):
    """
    Block 类是一个通用的神经网络块，用于构建编码器-解码器架构中的模块。

    初始化参数:
    - in_channels (int): 输入通道数。
    - downsample_t (Callable, 可选): 下采样模块的构造函数，默认为 DownsampleItem。
    - upsample_t (Callable, 可选): 上采样模块的构造函数，默认为 UpsampleItem。
    - skip_t (Callable, 可选): 跳跃连接模块的构造函数，默认为 SkipAdd。
    - skip_adapter_t (Callable, 可选): 跳跃连接适配器的构造函数，默认为 SkipAdapter。
    - items (Sequence[Callable], 可选): 下采样和中间处理模块的列表。
    - items_up (Optional[Sequence[Callable]], 可选): 上采样模块的列表。如果未提供，则使用 items。
    - out_channels (Optional[int], 可选): 输出通道数。如果未提供，则默认为 in_channels。
    - inner_block (Optional[nn.Module], 可选): 内部块模块，用于更复杂的嵌套结构。
    - **kwargs: 其他传递给模块构造函数的参数。
    """
    def __init__(
        self,
        in_channels: int,
        downsample_t: Callable = DownsampleItem,
        upsample_t: Callable = UpsampleItem,
        skip_t: Callable = SkipAdd,
        skip_adapter_t: Callable = SkipAdapter,
        items: Sequence[Callable] = [],
        items_up: Optional[Sequence[Callable]] = None,
        out_channels: Optional[int] = None,
        inner_block: Optional[nn.Module] = None,
        **kwargs,
    ):
        super().__init__()
        # 如果未提供 out_channels，则默认为 in_channels
        out_channels = default(out_channels, in_channels)

        # 如果未提供 items_up，则使用 items
        items_up = default(items_up, items)  # type: ignore
        # 构建下采样模块列表，先添加下采样模块，再添加中间处理模块
        items_down = [downsample_t] + list(items)
        # 构建上采样模块列表，先添加中间处理模块，再添加上采样模块
        items_up = list(items_up) + [upsample_t]
        # 构建传递给每个模块的关键字参数
        items_kwargs = dict(
            in_channels=in_channels, out_channels=out_channels, **kwargs
        )

        # Build items stack: items down -> inner block -> items up
        # 构建模块列表：下采样模块 -> 内部块 -> 上采样模块
        items_all: List[nn.Module] = []
        # 添加下采样模块
        items_all += [item_t(**items_kwargs) for item_t in items_down]
        # 添加内部块（如果存在）
        items_all += [inner_block] if exists(inner_block) else []
        # 添加上采样模块
        items_all += [item_t(**items_kwargs) for item_t in items_up]

        # 初始化跳跃连接适配器
        self.skip_adapter = skip_adapter_t(**items_kwargs)
        # 初始化包含所有模块的 Sequential 块
        self.block = Sequential(*items_all)
        # 初始化跳跃连接模块
        self.skip = skip_t(**items_kwargs)

    def forward(
        self,
        x: Tensor,
        features: Optional[Tensor] = None,
        embedding: Optional[Tensor] = None,
        channels: Optional[Sequence[Tensor]] = None,
    ) -> Tensor:
        """
        前向传播方法。

        参数:
        - x (Tensor): 输入张量。
        - features (Optional[Tensor], 可选): 特征张量。
        - embedding (Optional[Tensor], 可选): 嵌入张量。
        - channels (Optional[Sequence[Tensor]], 可选): 通道序列。

        返回:
        - Tensor: 输出张量。
        """
        # 应用跳跃连接适配器
        skip = self.skip_adapter(x)
        # 通过块处理输入张量
        x = self.block(x, features, embedding, channels)
        # 应用跳跃连接
        x = self.skip(skip, x, features)
        return x


# Block type, to be provided in UNet
# Block 类型，用于在 UNet 中提供
XBlock = T(Block, override=False)



################################################ UNet ################################################ 


class XUNet(nn.Module):
    """
    XUNet 类是一个通用的 UNet 模型，支持灵活的块配置。

    初始化参数:
    - in_channels (int): 输入通道数。
    - blocks (Sequence): 块的序列，每个块可以是 Block 或其他自定义块。
    - out_channels (Optional[int], 可选): 输出通道数。如果未提供，则默认为 in_channels。
    - **kwargs: 其他传递给块构造函数的参数。
    """
    def __init__(
        self,
        in_channels: int,
        blocks: Sequence,
        out_channels: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        num_layers = len(blocks)
        out_channels = default(out_channels, in_channels)

        def Net(i: int) -> Optional[nn.Module]:
            if i == num_layers:
                return None  # noqa
            block_t = blocks[i] # 获取当前层的块类型
            # 计算当前层的输入通道数
            in_ch = in_channels if i == 0 else blocks[i - 1].channels
            # 计算当前层的输出通道数
            out_ch = out_channels if i == 0 else in_ch

            # 实例化当前层的块
            return block_t(
                in_channels=in_ch,
                out_channels=out_ch,
                depth=i,
                inner_block=Net(i + 1),
                **kwargs,
            )

        # 构建整个网络
        self.net = Net(0)

    def forward(
        self,
        x: Tensor,
        *,
        features: Optional[Tensor] = None,
        embedding: Optional[Tensor] = None,
        channels: Optional[Sequence[Tensor]] = None,
    ) -> Tensor:
        """
        前向传播方法。

        参数:
        - x (Tensor): 输入张量。
        - features (Optional[Tensor], 可选): 特征张量。
        - embedding (Optional[Tensor], 可选): 嵌入张量。
        - channels (Optional[Sequence[Tensor]], 可选): 通道序列。

        返回:
        - Tensor: 输出张量。
        """
        return self.net(x, features, embedding, channels)  # type: ignore
