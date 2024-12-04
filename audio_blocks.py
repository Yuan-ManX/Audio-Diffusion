from math import pi
from typing import Any, Callable, Optional, Sequence, Type, TypeVar, Union

import torch
import torch.nn.functional as F
from einops import pack, rearrange, reduce, repeat, unpack
from torch import Tensor, einsum, nn
from typing_extensions import TypeGuard

V = TypeVar("V")


"""
这段代码提供了一组工具，用于创建灵活且可复用的类型模板、类型检查机制、默认值处理以及模块化的神经网络组件。
具体来说：
T 类: 用于动态生成类型模板，支持参数覆盖。
Ts 函数: 创建一个类型模板，接受一组类型实例并实例化它们。
exists 函数: 用于类型检查，判断值是否为 None。
default 函数: 返回可选值或默认值。
Module 函数: 创建一个模块化的神经网络组件，支持自定义的前向传播逻辑。
这些工具可以大大提高代码的可读性和可维护性，特别是在需要处理复杂类型系统和构建深度学习模型时。

"""

class T:
    """Where the magic happens, builds a type template for a given type"""
    """
    T 类用于构建给定类型的类型模板。

    初始化参数:
    - t (Callable): 需要被调用的函数或可调用对象。
    - override (bool, 可选): 是否覆盖传入的参数。默认为 True。
    """

    def __init__(self, t: Callable, override: bool = True):
        self.t = t # 存储传入的可调用对象
        self.override = override # 存储是否覆盖参数的标志

    def __call__(self, *a, **ka):
        t, override = self.t, self.override # 解包可调用对象和覆盖标志

        class Inner:
            def __init__(self):
                # 存储位置参数
                self.args = a
                # 更新实例字典以存储关键字参数
                self.__dict__.update(**ka)

            def __call__(self, *b, **kb):
                if override:
                    # 如果覆盖标志为 True，则先传入 a 和 b 作为位置参数，ka 和 kb 作为关键字参数
                    return t(*(*a, *b), **{**ka, **kb})
                else:
                    # 如果覆盖标志为 False，则先传入 b 和 a 作为位置参数，kb 和 ka 作为关键字参数
                    return t(*(*b, *a), **{**kb, **ka})

        return Inner() # 返回内部类 Inner 的实例


def Ts(t: Callable[..., V]) -> Callable[..., Callable[..., V]]:
    """Builds a type template for a given type that accepts a list of instances"""
    """
    Ts 是一个装饰器工厂，用于创建一个类型模板，该模板接受一组类型实例并生成一个新的可调用对象。

    参数:
    - t (Callable[..., V]): 需要被调用的函数或可调用对象。

    返回:
    - 一个新的可调用对象，该对象在被调用时会实例化所有传入的类型并调用函数 t。
    """
    # 返回一个匿名函数，该函数实例化所有传入的类型并调用 t
    return lambda *types: lambda: t(*[tp() for tp in types])


def exists(val: Optional[V]) -> TypeGuard[V]:
    """
    exists 函数用于类型检查，判断一个可选值是否不为 None。

    参数:
    - val (Optional[V]): 需要检查的值。

    返回:
    - 如果 val 不为 None，则返回 True；否则返回 False。
    """

    return val is not None


def default(val: Optional[V], d: V) -> V:
    """
    default 函数用于返回可选值，如果可选值为 None，则返回默认值。

    参数:
    - val (Optional[V]): 需要检查的可选值。
    - d (V): 默认值。

    返回:
    - 如果 val 存在，则返回 val；否则返回 d。
    """

    return val if exists(val) else d


def Module(modules: Sequence[nn.Module], forward_fn: Callable):
    """Functional module helper"""
    """
    Module 函数是一个辅助函数，用于创建一个模块化的神经网络组件。

    参数:
    - modules (Sequence[nn.Module]): 一组神经网络模块。
    - forward_fn (Callable): 前向传播函数，定义如何处理输入数据。

    返回:
    - 一个自定义的 nn.Module 实例。
    """

    class Module(nn.Module):
        def __init__(self):
            super().__init__()
            # 将传入的模块列表转换为 nn.ModuleList 并存储
            self.blocks = nn.ModuleList(modules)

        def forward(self, *args, **kwargs):
            # 调用传入的前向传播函数
            return forward_fn(*args, **kwargs)

    # 返回 Module 类的实例
    return Module()



"""
Sequential 类是一个自定义的顺序容器，用于将多个神经网络模块按顺序连接起来。
与 PyTorch 内置的 nn.Sequential 不同，此自定义版本允许在每个模块的前向传播过程中传递额外的参数。

继承自:
nn.Module: PyTorch 的基础模块类，所有自定义模块都应继承自它。

初始化参数:
*blocks: 任意数量的神经网络模块，这些模块将按顺序连接。

主要组件:
self.blocks: 使用 nn.ModuleList 存储传入的模块列表。


详细步骤:
1.初始化:
接受任意数量的模块作为位置参数 blocks，并使用 nn.ModuleList 将它们存储为 self.blocks。

2.前向传播方法 (forward):
接受一个输入张量 x 和任意数量的额外参数 *args。
遍历 self.blocks 中的每个模块：
将当前输入 x 和所有额外参数 *args 传递给当前模块。
将模块的输出赋值给 x，以便传递给下一个模块。
返回最后一个模块的输出。

"""

class Sequential(nn.Module):
    """Custom Sequential that includes all args"""

    def __init__(self, *blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: Tensor, *args) -> Tensor:
        for block in self.blocks:
            x = block(x, *args)
        return x


def Select(args_fn: Callable) -> Callable[..., Type[nn.Module]]:
    """Selects (swap, remove, repeat) forward arguments given a (lambda) function"""
    """
    Select 装饰器用于创建一个自定义的 nn.Module，该模块可以在前向传播过程中选择性地处理参数。

    参数:
    - args_fn (Callable): 一个函数，用于定义如何处理前向传播中的参数。该函数接受前向传播的参数，并返回一个新的参数元组。
                         例如，可以用于交换、移除或重复参数。

    返回:
    - 一个装饰器函数，该装饰器函数接受一个 nn.Module 类型并返回一个新的 nn.Module 类型。
    """

    def fn(block_t: Type[nn.Module]) -> Type[nn.Module]:
        """
        内部函数，用于创建一个 Select 类，该类继承自 nn.Module。

        参数:
        - block_t (Type[nn.Module]): 需要被包装的原始 nn.Module 类型。

        返回:
        - 一个新的 nn.Module 类型，该类型在调用时会使用 args_fn 处理参数。
        """

        class Select(nn.Module):
            """
                初始化 Select 模块。

                参数:
                - *args: 传递给原始模块 block_t 的位置参数。
                - **kwargs: 传递给原始模块 block_t 的关键字参数。

                初始化步骤:
                - 调用父类 nn.Module 的初始化方法。
                - 实例化原始模块 block_t，并将其存储为 self.block。
                - 存储传入的 args_fn 函数，以便在前向传播中使用。
                """
            
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.block = block_t(*args, **kwargs) # 实例化原始模块
                self.args_fn = args_fn

            def forward(self, *args, **kwargs):
                """
                前向传播方法。

                参数:
                - *args: 传递给模块的前向传播的位置参数。
                - **kwargs: 传递给模块的前向传播的关键字参数。

                处理步骤:
                - 使用 args_fn 函数处理传入的参数 args。
                - 将处理后的参数传递给原始模块 self.block。
                - 返回 self.block 的输出。
                """
                
                return self.block(*args_fn(*args), **kwargs)

        return Select

    return fn


class Packed(Sequential):
    """Packs, and transposes non-channel dims, useful for attention-like view"""
    """
    Packed 类继承自 Sequential，用于打包并转置非通道维度，适用于类似注意力机制的操作。

    前向传播方法:
    - 打包输入张量 x 的非通道维度。
    - 重塑张量形状以适应注意力机制的计算。
    - 通过 Sequential 模块处理张量。
    - 恢复原始张量的形状。
    - 返回处理后的张量。
    """

    def forward(self, x: Tensor, *args) -> Tensor:
        # 使用 einops 的 pack 函数将张量 x 打包，"b d *" 表示批次维度 b，通道维度 d 和其他维度 *
        x, ps = pack([x], "b d *")
        # 重塑张量形状，从 "b d n" 到 "b n d"，以便进行注意力计算
        x = rearrange(x, "b d n -> b n d")
        # 将重塑后的张量传递给 Sequential 模块进行处理
        x = super().forward(x, *args)
        # 重塑张量形状，从 "b n d" 恢复为原始的 "b d n"
        x = rearrange(x, "b n d -> b d n")
        # 使用 einops 的 unpack 函数解包张量，恢复打包前的形状
        x = unpack(x, ps, "b d *")[0]
        return x


def Repeat(m: Union[nn.Module, Type[nn.Module]], times: int) -> Any:
    """
    Repeat 函数用于重复一个模块多次，并将其封装在一个 Sequential 模块中。

    参数:
    - m (nn.Module 或 Type[nn.Module]): 需要重复的模块或模块类型。
    - times (int): 重复次数。

    返回:
    - 一个 Sequential 模块，包含重复的模块。
    """

    # 重复模块 m，生成一个包含 times 个模块的元组
    ms = (m,) * times
    # 如果 m 是 nn.Module 的实例，则直接使用 Sequential 封装
    # 否则，使用 Ts(Sequential) 装饰器封装（用于类型模板）
    return Sequential(*ms) if isinstance(m, nn.Module) else Ts(Sequential)(*ms)


def Skip(merge_fn: Callable[[Tensor, Tensor], Tensor] = torch.add) -> Type[Sequential]:
    """
    Skip 函数用于创建一个带有跳跃连接的 Sequential 模块。

    参数:
    - merge_fn (Callable[[Tensor, Tensor], Tensor], 可选): 用于合并主路径和跳跃连接路径的函数，默认为 torch.add。

    返回:
    - 一个新的 Sequential 类，包含跳跃连接逻辑。
    """

    class Skip(Sequential):

        """Adds skip connection around modules"""
        """
        Skip 类继承自 Sequential，添加了跳跃连接功能。

        前向传播方法:
        - 通过 Sequential 模块处理输入张量。
        - 使用 merge_fn 将原始输入张量与 Sequential 模块的输出张量合并。
        - 返回合并后的张量。
        """

        def forward(self, x: Tensor, *args) -> Tensor:
            return merge_fn(x, super().forward(x, *args)) # 合并主路径和跳跃连接路径

    return Skip



################################################ Modules ################################################


# 定义一个辅助函数，用于将整数维度映射到对应的卷积类
def Conv(dim: int, *args, **kwargs) -> nn.Module:
    """
    根据给定的维度选择并返回相应的卷积层。

    参数:
    - dim (int): 数据的维度（1, 2 或 3）。
    - *args: 传递给卷积层的其他位置参数。
    - **kwargs: 传递给卷积层的其他关键字参数。

    返回:
    - 对应维度的卷积层（nn.Conv1d, nn.Conv2d 或 nn.Conv3d）。
    """
     
    return [nn.Conv1d, nn.Conv2d, nn.Conv3d][dim - 1](*args, **kwargs)


# 定义一个辅助函数，用于将整数维度映射到对应的转置卷积类
def ConvTranspose(dim: int, *args, **kwargs) -> nn.Module:
    """
    根据给定的维度选择并返回相应的转置卷积层。

    参数:
    - dim (int): 数据的维度（1, 2 或 3）。
    - *args: 传递给转置卷积层的其他位置参数。
    - **kwargs: 传递给转置卷积层的其他关键字参数。

    返回:
    - 对应维度的转置卷积层（nn.ConvTranspose1d, nn.ConvTranspose2d 或 nn.ConvTranspose3d）。
    """

    return [nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d][dim - 1](
        *args, **kwargs
    )


# 定义一个下采样模块，根据给定的维度进行下采样
def Downsample(
    dim: int, factor: int = 2, width: int = 1, conv_t=Conv, **kwargs
) -> nn.Module:
    """
    下采样模块，根据给定的下采样因子和卷积参数进行下采样。

    参数:
    - dim (int): 数据的维度（1, 2 或 3）。
    - factor (int, 可选): 下采样因子，默认为2。
    - width (int, 可选): 卷积核宽度，默认为1。
    - conv_t (Callable[..., nn.Module], 可选): 卷积层的构造函数，默认为 Conv 函数。
    - **kwargs: 其他传递给卷积层的关键字参数。

    返回:
    - 下采样模块（卷积层）。
    """

    # 如果下采样因子大于1，则使用指定的宽度，否则宽度为1
    width = width if factor > 1 else 1
    return conv_t(
        dim=dim,
        kernel_size=factor * width,  # 卷积核大小为下采样因子乘以宽度
        stride=factor,  # 步幅为下采样因子
        padding=(factor * width - factor) // 2,  # 计算填充大小以保持输出尺寸
        **kwargs,
    )


# 定义一个上采样模块，根据给定的维度进行上采样
def Upsample(
    dim: int,
    factor: int = 2,
    width: int = 1,
    conv_t=Conv,
    conv_tranpose_t=ConvTranspose,
    **kwargs,
) -> nn.Module:
    """
    上采样模块，根据给定的上采样因子和卷积参数进行上采样。

    参数:
    - dim (int): 数据的维度（1, 2 或 3）。
    - factor (int, 可选): 上采样因子，默认为2。
    - width (int, 可选): 卷积核宽度，默认为1。
    - conv_t (Callable[..., nn.Module], 可选): 卷积层的构造函数，默认为 Conv 函数。
    - conv_tranpose_t (Callable[..., nn.Module], 可选): 转置卷积层的构造函数，默认为 ConvTranspose 函数。
    - **kwargs: 其他传递给转置卷积层的关键字参数。

    返回:
    - 上采样模块（转置卷积层）。
    """

    # 如果上采样因子大于1，则使用指定的宽度，否则宽度为1
    width = width if factor > 1 else 1
    return conv_tranpose_t(
        dim=dim,
        kernel_size=factor * width, # 转置卷积核大小为上采样因子乘以宽度
        stride=factor, # 步幅为上采样因子
        padding=(factor * width - factor) // 2, # 计算填充大小以保持输出尺寸
        **kwargs,
    )


# 定义一个使用插值进行上采样的模块
def UpsampleInterpolate(
    dim: int,
    factor: int = 2,
    kernel_size: int = 3,
    mode: str = "nearest",
    conv_t=Conv,
    **kwargs,
) -> nn.Module:
    """
    使用插值进行上采样，并应用卷积层进行平滑处理。

    参数:
    - dim (int): 数据的维度（1, 2 或 3）。
    - factor (int, 可选): 上采样因子，默认为2。
    - kernel_size (int, 可选): 卷积核大小，默认为3。
    - mode (str, 可选): 插值模式，默认为 "nearest"。
    - conv_t (Callable[..., nn.Module], 可选): 卷积层的构造函数，默认为 Conv 函数。
    - **kwargs: 其他传递给卷积层的关键字参数。

    返回:
    - 上采样模块（包含插值和卷积层的 Sequential 模块）。
    """
    
    # 确保卷积核大小为奇数
    assert kernel_size % 2 == 1, "upsample kernel size must be odd" 
    
    return nn.Sequential(
        nn.Upsample(scale_factor=factor, mode=mode), # 使用插值进行上采样
        conv_t(
            dim=dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, **kwargs
        ), # 应用卷积层进行平滑处理
    )


# 定义一个卷积块，包含归一化、激活函数和卷积层
def ConvBlock(
    dim: int,
    in_channels: int,
    activation_t=nn.SiLU,
    norm_t=T(nn.GroupNorm)(num_groups=1),
    conv_t=Conv,
    **kwargs,
) -> nn.Module:
    """
    卷积块模块，包含归一化、激活函数和卷积层。

    参数:
    - dim (int): 数据的维度（1, 2 或 3）。
    - in_channels (int): 输入通道数。
    - activation_t (Callable[[], nn.Module], 可选): 激活函数构造函数，默认为 nn.SiLU。
    - norm_t (Callable[[int], nn.Module], 可选): 归一化层构造函数，默认为 GroupNorm，组数为1。
    - conv_t (Callable[..., nn.Module], 可选): 卷积层的构造函数，默认为 Conv 函数。
    - **kwargs: 其他传递给卷积层的关键字参数。

    返回:
    - 卷积块（包含归一化、激活函数和卷积层的 Sequential 模块）。
    """

    return nn.Sequential(
        norm_t(num_channels=in_channels), # 应用归一化层
        activation_t(), # 应用激活函数
        conv_t(dim=dim, in_channels=in_channels, **kwargs), # 应用卷积层
    )


# 定义一个残差块，包含两个卷积块和一个跳跃连接
def ResnetBlock(
    dim: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    conv_block_t=ConvBlock,
    conv_t=Conv,
    **kwargs,
) -> nn.Module:
    """
    残差块模块，包含两个卷积块和一个跳跃连接。

    参数:
    - dim (int): 数据的维度（1, 2 或 3）。
    - in_channels (int): 输入通道数。
    - out_channels (int): 输出通道数。
    - kernel_size (int, 可选): 卷积核大小，默认为3。
    - conv_block_t (Callable[..., nn.Module], 可选): 卷积块构造函数，默认为 ConvBlock。
    - conv_t (Callable[..., nn.Module], 可选): 卷积层的构造函数，默认为 Conv 函数。
    - **kwargs: 其他传递给卷积块和卷积层的关键字参数。

    返回:
    - 残差块（包含两个卷积块和一个跳跃连接的 Sequential 模块）。
    """

    # 初始化卷积块，卷积核大小为 kernel_size，填充为 (kernel_size - 1) // 2
    ConvBlock = T(conv_block_t)(
        dim=dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, **kwargs
    )
    # 初始化 1x1 卷积层，用于调整通道数
    Conv = T(conv_t)(dim=dim, kernel_size=1)

    # 创建两个卷积块
    conv_block = Sequential(
        ConvBlock(in_channels=in_channels, out_channels=out_channels),
        ConvBlock(in_channels=out_channels, out_channels=out_channels),
    )

    # 如果输入和输出通道数不同，则使用 1x1 卷积层进行通道数匹配
    conv = nn.Identity()
    if in_channels != out_channels:
        conv = Conv(in_channels=in_channels, out_channels=out_channels)

    # 返回残差块，包含卷积块和跳跃连接
    return Module([conv_block, conv], lambda x: conv_block(x) + conv(x))


# 定义 GRN（全局响应归一化）层，通用适用于任意维度
class GRN(nn.Module):
    """GRN (Global Response Normalization) layer from ConvNextV2 generic to any dim"""
    """
    GRN（全局响应归一化）层，来自 ConvNextV2，适用于任意维度。

    初始化参数:
    - dim (int): 数据的维度。
    - channels (int): 通道数。
    """

    def __init__(self, dim: int, channels: int):
        super().__init__()
        # 创建一个全1元组，用于初始化 gamma 和 beta 参数的形状
        ones = (1,) * dim
        # 初始化 gamma 参数，形状为 (1, channels, *ones)
        self.gamma = nn.Parameter(torch.zeros(1, channels, *ones))
        # 初始化 beta 参数，形状为 (1, channels, *ones)
        self.beta = nn.Parameter(torch.zeros(1, channels, *ones))
        # 定义需要归一化的维度，范围从 2 到 dim + 1
        self.norm_dims = [d + 2 for d in range(dim)]

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播方法。

        参数:
        - x (Tensor): 输入张量，形状为 (batch_size, channels, ...)。

        返回:
        - Tensor: 经过 GRN 处理后的输出张量。
        """

        # 计算输入张量 x 的 L2 范数，维度为 self.norm_dims，保持维度不变
        Gx = torch.norm(x, p=2, dim=self.norm_dims, keepdim=True)
        # 计算归一化后的值 Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        # 应用 GRN 公式：gamma * (x * Nx) + beta + x
        return self.gamma * (x * Nx) + self.beta + x


# 定义 ConvNextV2 块，通用适用于任意维度
def ConvNextV2Block(dim: int, channels: int) -> nn.Module:
    """
    定义 ConvNextV2 块，包含深度卷积、组归一化、逐点卷积、激活函数和 GRN。

    参数:
    - dim (int): 数据的维度。
    - channels (int): 输入和输出的通道数。

    返回:
    - nn.Module: ConvNextV2 块。
    """

    block = nn.Sequential(
        # Depthwise and LayerNorm
        # 深度卷积和 LayerNorm
        Conv(
            dim=dim,
            in_channels=channels,
            out_channels=channels,
            kernel_size=7,
            padding=3,
            groups=channels, # 深度卷积，组数等于通道数
        ),
        nn.GroupNorm(num_groups=1, num_channels=channels),  # 组归一化
        # Pointwise expand
        # 逐点扩展卷积
        Conv(dim=dim, in_channels=channels, out_channels=channels * 4, kernel_size=1),
        # Activation and GRN
        nn.GELU(),
        GRN(dim=dim, channels=channels * 4),
        # Pointwise contract
        # 逐点收缩卷积
        Conv(
            dim=dim,
            in_channels=channels * 4,
            out_channels=channels,
            kernel_size=1,
        ),
    )

    # 使用残差连接，将输入与卷积块的输出相加
    return Module([block], lambda x: x + block(x))


# 定义基础注意力机制
def AttentionBase(features: int, head_features: int, num_heads: int) -> nn.Module:
    """
    基础注意力机制的实现。

    参数:
    - features (int): 输入特征的维度。
    - head_features (int): 每个注意力头的特征维度。
    - num_heads (int): 注意力头的数量。

    返回:
    - nn.Module: 注意力机制模块。
    """

    # 缩放因子
    scale = head_features**-0.5
    # 中间特征的维度
    mid_features = head_features * num_heads
    # 输出线性层
    to_out = nn.Linear(in_features=mid_features, out_features=features, bias=False)

    def forward(
        q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        h = num_heads
        # Split heads
        # 分割注意力头的维度
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        # Compute similarity matrix and add eventual mask
        # 计算相似度矩阵并应用掩码
        sim = einsum("... n d, ... m d -> ... n m", q, k) * scale
        # Get attention matrix with softmax
        # 应用 softmax 计算注意力权重
        attn = sim.softmax(dim=-1)
        # Compute values
        # 计算输出值
        out = einsum("... n m, ... m d -> ... n d", attn, v)
        # 重塑输出形状
        out = rearrange(out, "b h n d -> b n (h d)")
        return to_out(out)

    return Module([to_out], forward)


# 定义线性注意力机制
def LinearAttentionBase(features: int, head_features: int, num_heads: int) -> nn.Module:
    """
    线性注意力机制的实现。

    参数:
    - features (int): 输入特征的维度。
    - head_features (int): 每个注意力头的特征维度。
    - num_heads (int): 注意力头的数量。

    返回:
    - nn.Module: 线性注意力机制模块。
    """

    scale = head_features**-0.5 # 缩放因子
    mid_features = head_features * num_heads # 中间特征的维度
    to_out = nn.Linear(in_features=mid_features, out_features=features, bias=False) # 输出线性层

    def forward(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        h = num_heads
        # Split heads
        # 分割注意力头的维度
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        # Softmax rows and cols
        # 对 q 和 k 应用 softmax 归一化
        q = q.softmax(dim=-1) * scale
        k = k.softmax(dim=-2)
        # Attend on channel dim
        # 计算注意力权重
        attn = einsum("... n d, ... n c -> ... d c", k, v)
        # 计算输出值
        out = einsum("... n d, ... d c -> ... n c", q, attn)
        # 重塑输出形状
        out = rearrange(out, "b h n d -> b n (h d)")
        return to_out(out)

    return Module([to_out], forward)


# 定义固定位置嵌入
def FixedEmbedding(max_length: int, features: int):
    """
    生成固定的位置嵌入。

    参数:
    - max_length (int): 最大序列长度。
    - features (int): 嵌入的维度。

    返回:
    - nn.Module: 位置嵌入模块。
    """

    # 初始化嵌入层
    embedding = nn.Embedding(max_length, features)

    def forward(x: Tensor) -> Tensor:
        # 获取输入张量的批次大小和序列长度
        batch_size, length, device = *x.shape[0:2], x.device
        # 确保输入序列长度不超过最大长度
        assert_message = "Input sequence length must be <= max_length"
        assert length <= max_length, assert_message
        # 生成位置索引
        position = torch.arange(length, device=device)
        # 生成固定的位置嵌入
        fixed_embedding = embedding(position)
        # 重复位置嵌入以匹配批次大小
        fixed_embedding = repeat(fixed_embedding, "n d -> b n d", b=batch_size)
        return fixed_embedding

    return Module([embedding], forward)


class Attention(nn.Module):
    """
    自定义注意力机制模块，支持多头注意力、条件上下文和位置嵌入。

    初始化参数:
    - features (int): 输入特征的维度。
    - head_features (int): 每个注意力头的特征维度。
    - num_heads (int): 注意力头的数量。
    - context_features (Optional[int], 可选): 上下文特征的维度。如果提供，则使用条件上下文。
    - max_length (Optional[int], 可选): 最大序列长度，用于位置嵌入。如果使用位置嵌入，则必须提供。
    - attention_base_t (Callable, 可选): 注意力机制的基础类，默认为 AttentionBase。
    - positional_embedding_t (Optional[Callable], 可选): 位置嵌入的构造函数。如果提供，则使用位置嵌入。
    """

    def __init__(
        self,
        features: int,
        *,
        head_features: int,
        num_heads: int,
        context_features: Optional[int] = None,
        max_length: Optional[int] = None,
        attention_base_t=AttentionBase,
        positional_embedding_t=None,
    ):
        super().__init__()
        # 存储上下文特征的维度
        self.context_features = context_features
        # 是否使用位置嵌入
        self.use_positional_embedding = exists(positional_embedding_t) 
        # 是否使用条件上下文
        self.use_context = exists(context_features)
        # 中间特征的维度
        mid_features = head_features * num_heads
        # 如果未提供上下文特征，则默认为输入特征的维度
        context_features = default(context_features, features)

        # 存储最大序列长度
        self.max_length = max_length
        if self.use_positional_embedding:
            assert exists(max_length)
            # 初始化位置嵌入层
            self.positional_embedding = positional_embedding_t(
                max_length=max_length, features=features
            )

        # 初始化层归一化层，用于输入特征
        self.norm = nn.LayerNorm(features)
        # 初始化层归一化层，用于上下文特征
        self.norm_context = nn.LayerNorm(context_features)
        # 初始化线性层，将输入特征映射到中间特征，用于查询（q）
        self.to_q = nn.Linear(
            in_features=features, out_features=mid_features, bias=False
        )
        # 初始化线性层，将上下文特征映射到中间特征，用于键（k）和值（v）
        self.to_kv = nn.Linear(
            in_features=context_features, out_features=mid_features * 2, bias=False
        )
        # 初始化注意力机制模块
        self.attention = attention_base_t(
            features, num_heads=num_heads, head_features=head_features
        )

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        """
        前向传播方法。

        参数:
        - x (Tensor): 输入张量，形状为 (batch_size, sequence_length, features)。
        - context (Optional[Tensor], 可选): 上下文张量，形状为 (batch_size, context_length, context_features)。

        返回:
        - Tensor: 输出张量，形状与输入 x 相同。
        """

        assert_message = "You must provide a context when using context_features" 
        # 如果使用上下文特征，则必须提供 context
        assert not self.context_features or exists(context), assert_message

        # 保存输入张量作为跳跃连接
        skip = x

        if self.use_positional_embedding:
            # 如果使用位置嵌入，则将位置嵌入添加到输入张量 x 中
            x = x + self.positional_embedding(x)

        # Use context if provided
        # 如果提供了 context 并且使用上下文，则使用 context；否则，使用 x 作为上下文
        context = context if exists(context) and self.use_context else x
        # Normalize then compute q from input and k,v from context
        # 对输入张量 x 和上下文张量 context 应用层归一化
        x, context = self.norm(x), self.norm_context(context)

        # 将输入张量 x 通过线性层转换为查询 q, 将上下文张量 context 通过线性层转换为键 k 和值 v
        q, k, v = (self.to_q(x), *torch.chunk(self.to_kv(context), chunks=2, dim=-1))
        # Compute and return attention
        # 计算注意力并返回跳跃连接后的结果
        return skip + self.attention(q, k, v)


def CrossAttention(context_features: int, **kwargs):
    """
    交叉注意力机制。

    参数:
    - context_features (int): 上下文特征的维度。
    - **kwargs: 其他传递给 Attention 的关键字参数。

    返回:
    - nn.Module: 交叉注意力模块。
    """

    return Attention(context_features=context_features, **kwargs)


def FeedForward(features: int, multiplier: int) -> nn.Module:
    """
    前馈网络（FeedForward Network）。

    参数:
    - features (int): 输入特征的维度。
    - multiplier (int): 中间特征的倍数。

    返回:
    - nn.Module: 前馈网络模块。
    """

    # 计算中间特征的维度
    mid_features = features * multiplier
    return Skip(torch.add)(
        # 使用 Skip 封装器添加跳跃连接
        # 第一个线性层
        nn.Linear(in_features=features, out_features=mid_features),
        # GELU 激活函数
        nn.GELU(),
        # 第二个线性层
        nn.Linear(in_features=mid_features, out_features=features),
    )


def Modulation(in_features: int, num_features: int) -> nn.Module:
    """
    调制模块，用于缩放和偏移输入特征。

    参数:
    - in_features (int): 输入特征的维度。
    - num_features (int): 调制特征的维度。

    返回:
    - nn.Module: 调制模块。
    """

    to_scale_shift = nn.Sequential(
        nn.SiLU(),
        # 线性层，输出维度为 in_features * 2
        nn.Linear(in_features=num_features, out_features=in_features * 2, bias=True),
    )
    # 层归一化，不使用仿射变换
    norm = nn.LayerNorm(in_features, elementwise_affine=False, eps=1e-6)

    def forward(x: Tensor, features: Tensor) -> Tensor:
        """
        前向传播方法。

        参数:
        - x (Tensor): 输入张量。
        - features (Tensor): 调制特征。

        返回:
        - Tensor: 调制后的输出张量。
        """
        # 计算缩放和偏移参数
        scale_shift = to_scale_shift(features)
        # 拆分缩放和偏移
        scale, shift = rearrange(scale_shift, "b d -> b 1 d").chunk(2, dim=-1)
        # 应用调制
        return norm(x) * (1 + scale) + shift

    return Module([to_scale_shift, norm], forward)


def MergeAdd():
    """
    合并函数，简单的元素级相加。

    返回:
    - nn.Module: 合并模块，执行元素级相加。
    """
    return Module([], lambda x, y, *_: x + y)


def MergeCat(dim: int, channels: int, scale: float = 2**-0.5) -> nn.Module:
    """
    合并函数，连接两个张量并应用卷积层。

    参数:
    - dim (int): 数据维度。
    - channels (int): 通道数。
    - scale (float, 可选): 缩放因子，默认为 2**-0.5。

    返回:
    - nn.Module: 合并模块，执行连接和卷积操作。
    """
    conv = Conv(dim=dim, in_channels=channels * 2, out_channels=channels, kernel_size=1)
    return Module([conv], lambda x, y, *_: conv(torch.cat([x * scale, y], dim=1)))


def MergeModulate(dim: int, channels: int, modulation_features: int):
    """
    合并并调制函数，连接两个张量并进行调制。

    参数:
    - dim (int): 数据维度。
    - channels (int): 通道数。
    - modulation_features (int): 调制特征的维度。

    返回:
    - nn.Module: 合并并调制模块。
    """
    to_scale = nn.Sequential(
        nn.SiLU(),
        nn.Linear(in_features=modulation_features, out_features=channels, bias=True),
    )

    def forward(x: Tensor, y: Tensor, features: Tensor, *args) -> Tensor:
        """
        前向传播方法。

        参数:
        - x (Tensor): 第一个输入张量。
        - y (Tensor): 第二个输入张量。
        - features (Tensor): 调制特征。

        返回:
        - Tensor: 合并并调制后的输出张量。
        """
        # 调整缩放因子形状
        scale = rearrange(to_scale(features), f'b c -> b c {"1 " * dim}')
        # 合并并调制
        return x + scale * y

    return Module([to_scale], forward)



################################################ Embedders ################################################


class NumberEmbedder(nn.Module):
    """
    NumberEmbedder 类用于将数字嵌入到高维空间中。

    初始化参数:
    - features (int): 输出嵌入特征的维度。
    - dim (int, 可选): 内部表示的维度，默认为256。必须能被2整除。
    """
    def __init__(self, features: int, dim: int = 256):
        super().__init__()
        assert dim % 2 == 0, f"dim must be divisible by 2, found {dim}" # 确保 dim 是偶数
        self.features = features # 输出特征的维度
        self.weights = nn.Parameter(torch.randn(dim // 2)) # 初始化权重参数，形状为 (dim/2,)
        self.to_out = nn.Linear(in_features=dim + 1, out_features=features) # 线性层，将输入维度 dim+1 映射到输出维度 features

    def to_embedding(self, x: Tensor) -> Tensor:
        """
        将输入张量 x 转换为嵌入向量。

        参数:
        - x (Tensor): 输入张量，形状为 (batch_size, 1)。

        返回:
        - Tensor: 嵌入后的张量，形状为 (batch_size, features)。
        """
        x = rearrange(x, "b -> b 1") # 重塑张量形状为 (batch_size, 1)
        # 计算频率，形状为 (batch_size, dim/2)
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * pi
        # 将正弦和余弦结果连接起来，形状为 (batch_size, dim)
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        # 将原始输入 x 和傅里叶变换后的结果连接起来，形状为 (batch_size, dim + 1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        # 通过线性层输出嵌入向量，形状为 (batch_size, features)
        return self.to_out(fouriered)

    def forward(self, x: Union[Sequence[float], Tensor]) -> Tensor:
        """
        前向传播方法，将输入数据转换为嵌入向量。

        参数:
        - x (Union[Sequence[float], Tensor]): 输入数据，可以是浮点数序列或张量。

        返回:
        - Tensor: 嵌入后的张量，形状为 (..., features)。
        """
        if not torch.is_tensor(x):
            # 如果输入不是张量，则转换为张量
            x = torch.tensor(x, device=self.weights.device)
        assert isinstance(x, Tensor) # 确保输入是张量
        # 获取输入张量的形状
        shape = x.shape
        x = rearrange(x, "... -> (...)") # 重塑张量形状为 (batch_size, ...)
        # 将嵌入向量重塑为原始形状并添加特征维度，形状为 (..., features)
        return self.to_embedding(x).view(*shape, self.features)  # type: ignore


class T5Embedder(nn.Module):
    """
    T5Embedder 类使用预训练的 T5 模型将文本转换为嵌入向量。

    初始化参数:
    - model (str, 可选): T5 模型的名称，默认为 "t5-base"。
    - max_length (int, 可选): 文本的最大长度，默认为64。
    """
    def __init__(self, model: str = "t5-base", max_length: int = 64):
        super().__init__()
        # 从 transformers 库中导入 AutoTokenizer 和 T5EncoderModel
        from transformers import AutoTokenizer, T5EncoderModel

        # 加载预训练的 T5 分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        # 加载预训练的 T5 编码器模型
        self.transformer = T5EncoderModel.from_pretrained(model)
        # 设置最大文本长度
        self.max_length = max_length
        # 获取模型的嵌入特征维度
        self.embedding_features = self.transformer.config.d_model

    @torch.no_grad()
    def forward(self, texts: Sequence[str]) -> Tensor:
        """
        前向传播方法，将文本转换为嵌入向量。

        参数:
        - texts (Sequence[str]): 输入的文本序列。

        返回:
        - Tensor: 文本的嵌入向量，形状为 (batch_size, sequence_length, embedding_features)。
        """
        encoded = self.tokenizer(
            texts,
            truncation=True, # 截断文本以适应最大长度
            max_length=self.max_length, # 设置最大长度
            padding="max_length", # 填充文本以达到最大长度
            return_tensors="pt", # 返回 PyTorch 张量
        )

        device = next(self.transformer.parameters()).device # 获取模型的设备
        input_ids = encoded["input_ids"].to(device) # 将输入 IDs 移动到模型设备
        attention_mask = encoded["attention_mask"].to(device)  # 将注意力掩码移动到模型设备

        self.transformer.eval() # 设置模型为评估模式

        embedding = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask
        )["last_hidden_state"] # 获取模型的最后一层隐藏状态作为嵌入向量
        
        # 返回嵌入向量
        return embedding



################################################ Plugins ################################################


def rand_bool(shape: Any, proba: float, device: Any = None) -> Tensor:
    """
    生成一个随机布尔张量。

    参数:
    - shape (Any): 张量的形状。
    - proba (float): 元素为 True 的概率。
    - device (Any, 可选): 张量所在的设备。默认为 None。

    返回:
    - Tensor: 生成的布尔张量，形状为 `shape`，元素为 True 或 False。
    """
    if proba == 1:
        # 如果概率为1，则返回全1张量
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif proba == 0:
        # 如果概率为0，则返回全0张量
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        # 否则，使用伯努利分布生成随机布尔张量
        return torch.bernoulli(torch.full(shape, proba, device=device)).to(torch.bool)


def ClassifierFreeGuidancePlugin(
    net_t: Type[nn.Module],
    embedding_max_length: int,
) -> Callable[..., nn.Module]:
    """Classifier-Free Guidance -> CFG(UNet, embedding_max_length=512)(...)"""
    """
    Classifier-Free Guidance 插件，用于条件生成模型（如 UNet）。

    参数:
    - net_t (Type[nn.Module]): 原始模型类型。
    - embedding_max_length (int): 嵌入的最大长度。

    返回:
    - Callable[..., nn.Module]: 一个函数，返回带有 Classifier-Free Guidance 的模型。
    """

    def Net(embedding_features: int, **kwargs) -> nn.Module:
        """
        创建带有 Classifier-Free Guidance 的模型。

        参数:
        - embedding_features (int): 嵌入特征的维度。
        - **kwargs: 其他传递给原始模型的参数。

        返回:
        - nn.Module: 带有 Classifier-Free Guidance 的模型。
        """
        # 初始化固定嵌入层
        fixed_embedding = FixedEmbedding(
            max_length=embedding_max_length,
            features=embedding_features,
        )
        # 实例化原始模型
        net = net_t(embedding_features=embedding_features, **kwargs)  # type: ignore

        def forward(
            x: Tensor,
            embedding: Optional[Tensor] = None,
            embedding_scale: float = 1.0,
            embedding_mask_proba: float = 0.0,
            **kwargs,
        ):
            """
            前向传播方法，应用 Classifier-Free Guidance。

            参数:
            - x (Tensor): 输入张量。
            - embedding (Optional[Tensor], 可选): 条件嵌入向量。
            - embedding_scale (float, 可选): 条件嵌入的缩放因子，默认为1.0。
            - embedding_mask_proba (float, 可选): 嵌入掩码的概率，默认为0.0。
            - **kwargs: 其他传递给模型的参数。

            返回:
            - Tensor: 输出张量。
            """
            msg = "ClassiferFreeGuidancePlugin requires embedding" # 确保提供了嵌入
            assert exists(embedding), msg
            b, device = embedding.shape[0], embedding.device  # 获取批次大小和设备
            embedding_mask = fixed_embedding(embedding) # 生成固定嵌入掩码
 
            if embedding_mask_proba > 0.0:
                # Randomly mask embedding
                # 如果嵌入掩码概率大于0，则随机掩码嵌入
                batch_mask = rand_bool(
                    shape=(b, 1, 1), proba=embedding_mask_proba, device=device
                )
                embedding = torch.where(batch_mask, embedding_mask, embedding) # 应用掩码

            if embedding_scale != 1.0:
                # Compute both normal and fixed embedding outputs
                # 如果嵌入缩放因子不为1，则应用 Classifier-Free Guidance
                out = net(x, embedding=embedding, **kwargs) # 条件输出
                out_masked = net(x, embedding=embedding_mask, **kwargs)  # 无条件输出
                # Scale conditional output using classifier-free guidance
                # 使用 Classifier-Free Guidance 公式计算最终输出
                return out_masked + (out - out_masked) * embedding_scale
            else:
                # 否则，直接返回条件输出
                return net(x, embedding=embedding, **kwargs)

        # 返回带有 Classifier-Free Guidance 的模型
        return Module([fixed_embedding, net], forward)

    return Net


def TimeConditioningPlugin(
    net_t: Type[nn.Module],
    num_layers: int = 2,
) -> Callable[..., nn.Module]:
    """Adds time conditioning (e.g. for diffusion)"""
    """
    TimeConditioningPlugin 用于为模型添加时间条件（如扩散模型）。

    参数:
    - net_t (Type[nn.Module]): 原始模型类型。
    - num_layers (int, 可选): MLP 的层数，默认为2。

    返回:
    - Callable[..., nn.Module]: 一个函数，返回带有时间条件的模型。
    """

    def Net(modulation_features: Optional[int] = None, **kwargs) -> nn.Module:
        """
        创建带有时间条件的模型。

        参数:
        - modulation_features (Optional[int], 可选): 调制特征的维度。
        - **kwargs: 其他传递给原始模型的参数。

        返回:
        - nn.Module: 带有时间条件的模型。
        """
        msg = "TimeConditioningPlugin requires modulation_features"
        assert exists(modulation_features), msg # 确保提供了调制特征维度

        embedder = NumberEmbedder(features=modulation_features) # 初始化数字嵌入器
        # 初始化 MLP
        mlp = Repeat(
            nn.Sequential(
                nn.Linear(modulation_features, modulation_features), nn.GELU()
            ),
            times=num_layers,
        )
        # 实例化原始模型
        net = net_t(modulation_features=modulation_features, **kwargs)  # type: ignore

        def forward(
            x: Tensor,
            time: Optional[Tensor] = None,
            features: Optional[Tensor] = None,
            **kwargs,
        ):
            msg = "TimeConditioningPlugin requires time in forward"
            assert exists(time), msg # 确保提供了时间信息

            # Process time to time_features
            # 将时间信息转换为时间特征
            time_features = F.gelu(embedder(time))
            # 通过 MLP 处理时间特征
            time_features = mlp(time_features)

            # Overlap features if more than one per batch
            # 如果时间特征有多个维度，则进行降维
            if time_features.ndim == 3:
                time_features = reduce(time_features, "b n d -> b d", "sum")

            # Merge time features with features if provided
            # 将时间特征与特征合并
            features = features + time_features if exists(features) else time_features
            return net(x, features=features, **kwargs)

        return Module([embedder, mlp, net], forward)

    return Net


def TextConditioningPlugin(
    net_t: Type[nn.Module], embedder: Optional[nn.Module] = None
) -> Callable[..., nn.Module]:
    """Adds text conditioning"""
    """
    TextConditioningPlugin 用于为模型添加文本条件。

    参数:
    - net_t (Type[nn.Module]): 原始模型类型。
    - embedder (Optional[nn.Module], 可选): 文本嵌入器。如果未提供，则使用默认的 T5Embedder。

    返回:
    - Callable[..., nn.Module]: 一个函数，返回带有文本条件的模型。
    """
    # 使用默认的 T5Embedder 或用户提供的嵌入器
    embedder = embedder if exists(embedder) else T5Embedder()
    msg = "TextConditioningPlugin embedder requires embedding_features attribute"
    # 确保嵌入器具有 embedding_features 属性
    assert hasattr(embedder, "embedding_features"), msg
    # 获取嵌入特征的维度
    features: int = embedder.embedding_features  # type: ignore

    def Net(embedding_features: int = features, **kwargs) -> nn.Module:
        """
        创建带有文本条件的模型。

        参数:
        - embedding_features (int, 可选): 嵌入特征的维度，默认为 `features`。
        - **kwargs: 其他传递给原始模型的参数。

        返回:
        - nn.Module: 带有文本条件的模型。
        """
        msg = f"TextConditioningPlugin requires embedding_features={features}"
        # 确保嵌入特征的维度正确
        assert embedding_features == features, msg
        # 实例化原始模型
        net = net_t(embedding_features=embedding_features, **kwargs)  # type: ignore

        def forward(
            x: Tensor, text: Sequence[str], embedding: Optional[Tensor] = None, **kwargs
        ):
            """
            前向传播方法，应用文本条件。

            参数:
            - x (Tensor): 输入张量。
            - text (Sequence[str]): 输入的文本序列。
            - embedding (Optional[Tensor], 可选): 其他嵌入向量。

            返回:
            - Tensor: 输出张量。
            """
            # 生成文本嵌入
            text_embedding = embedder(text)  # type: ignore
            if exists(embedding):
                # 如果存在其他嵌入，则与文本嵌入连接
                text_embedding = torch.cat([text_embedding, embedding], dim=1)
            return net(x, embedding=text_embedding, **kwargs)

        # 返回带有文本条件的模型
        return Module([embedder, net], forward)  # type: ignore

    return Net

