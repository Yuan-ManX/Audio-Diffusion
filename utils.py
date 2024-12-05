from functools import reduce
from inspect import isfunction
from math import ceil, floor, log2, pi
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Generator, Tensor
from typing_extensions import TypeGuard


# 定义一个类型变量 T，用于泛型编程
T = TypeVar("T")


# 定义 exists 函数，用于检查一个值是否存在（不为 None）
def exists(val: Optional[T]) -> TypeGuard[T]:
    """
    检查一个值是否存在（不为 None）。

    参数:
    - val (Optional[T]): 需要检查的值。

    返回:
    - TypeGuard[T]: 如果 val 不为 None，则返回 True，表示 val 的类型为 T。
    """
    return val is not None


# 定义 iff 函数，根据条件返回值或 None
def iff(condition: bool, value: T) -> Optional[T]:
    """
    根据条件返回值或 None。

    参数:
    - condition (bool): 条件表达式。
    - value (T): 如果条件为 True，则返回该值。

    返回:
    - Optional[T]: 如果条件为 True，则返回 value；否则返回 None。
    """
    return value if condition else None


# 定义 is_sequence 函数，检查一个对象是否为列表或元组
def is_sequence(obj: T) -> TypeGuard[Union[list, tuple]]:
    """
    检查一个对象是否为列表或元组。

    参数:
    - obj (T): 需要检查的对象。

    返回:
    - TypeGuard[Union[list, tuple]]: 如果 obj 是列表或元组，则返回 True。
    """
    return isinstance(obj, list) or isinstance(obj, tuple)


# 定义 default 函数，返回可选值或默认值
def default(val: Optional[T], d: Union[Callable[..., T], T]) -> T:
    """
    返回可选值或默认值。

    参数:
    - val (Optional[T]): 需要检查的可选值。
    - d (Union[Callable[..., T], T]): 默认值。如果 d 是可调用对象，则调用它以获取默认值。

    返回:
    - T: 如果 val 存在，则返回 val；否则返回 d 的值或调用 d() 的返回值。
    """
    if exists(val):
        return val
    return d() if isfunction(d) else d


# 定义 to_list 函数，将输入转换为列表
def to_list(val: Union[T, Sequence[T]]) -> List[T]:
    """
    将输入转换为列表。

    参数:
    - val (Union[T, Sequence[T]]): 输入值，可以是单个元素或序列。

    返回:
    - List[T]: 转换后的列表。
    """
    if isinstance(val, tuple):
        return list(val)
    if isinstance(val, list):
        return val
    return [val]  # type: ignore


# 定义 prod 函数，计算序列中所有整数的乘积
def prod(vals: Sequence[int]) -> int:
    """
    计算序列中所有整数的乘积。

    参数:
    - vals (Sequence[int]): 整数序列。

    返回:
    - int: 所有整数的乘积。
    """
    return reduce(lambda x, y: x * y, vals)


# 定义 closest_power_2 函数，找到最接近输入值的2的幂
def closest_power_2(x: float) -> int:
    """
    找到最接近输入值的2的幂。

    参数:
    - x (float): 输入值。

    返回:
    - int: 最接近 x 的2的幂。
    """
    # 计算 x 的以2为底的对数
    exponent = log2(x)
    # 定义距离函数，计算 x 与 2^z 的绝对差值
    distance_fn = lambda z: abs(x - 2 ** z)  # noqa
    # 找到最接近的整数指数
    exponent_closest = min((floor(exponent), ceil(exponent)), key=distance_fn)
    # 返回最接近的2的幂
    return 2 ** int(exponent_closest)



################################################ Kwargs Utils ################################################ 


# 定义 group_dict_by_prefix 函数，根据键是否以指定前缀开头，将字典分成两部分
def group_dict_by_prefix(prefix: str, d: Dict) -> Tuple[Dict, Dict]:
    """
    根据键是否以指定前缀开头，将字典分成两部分。

    参数:
    - prefix (str): 前缀字符串，用于判断键是否以此开头。
    - d (Dict): 输入字典。

    返回:
    - Tuple[Dict, Dict]: 返回一个元组，包含两个字典。
                         第一个字典包含不以 prefix 开头的键值对，
                         第二个字典包含以 prefix 开头的键值对。
    """
    return_dicts: Tuple[Dict, Dict] = ({}, {}) # 初始化两个空字典
    # 遍历输入字典的所有键
    for key in d.keys():
        # 判断键是否不以 prefix 开头，转换为整数（0 或 1）
        no_prefix = int(not key.startswith(prefix))
        # 根据判断结果，将键值对放入相应的字典中
        return_dicts[no_prefix][key] = d[key]
    # 返回分割后的两个字典    
    return return_dicts


# 定义 groupby 函数，根据键是否以指定前缀开头，将字典分成两部分，并可选择是否保留前缀
def groupby(prefix: str, d: Dict, keep_prefix: bool = False) -> Tuple[Dict, Dict]:
    """
    根据键是否以指定前缀开头，将字典分成两部分，并可选择是否保留前缀。

    参数:
    - prefix (str): 前缀字符串，用于判断键是否以此开头。
    - d (Dict): 输入字典。
    - keep_prefix (bool, 可选): 是否保留前缀。默认为 False。

    返回:
    - Tuple[Dict, Dict]: 返回一个元组，包含两个字典。
                         如果 keep_prefix 为 False，第一个字典的键不包含 prefix；
                         如果 keep_prefix 为 True，第一个字典的键保留 prefix。
                         第二个字典始终包含以 prefix 开头的键值对。
    """
    # 使用 group_dict_by_prefix 分割字典
    kwargs_with_prefix, kwargs = group_dict_by_prefix(prefix, d)
    if keep_prefix:
        # 如果保留前缀，则返回分割后的字典
        return kwargs_with_prefix, kwargs
    # 否则，移除键中的前缀
    kwargs_no_prefix = {k[len(prefix) :]: v for k, v in kwargs_with_prefix.items()}
    # 返回移除前缀后的字典和原始字典
    return kwargs_no_prefix, kwargs


# 定义 prefix_dict 函数，为字典的所有键添加指定前缀
def prefix_dict(prefix: str, d: Dict) -> Dict:
    """
    为字典的所有键添加指定前缀。

    参数:
    - prefix (str): 前缀字符串。
    - d (Dict): 输入字典。

    返回:
    - Dict: 返回一个新字典，所有键都添加了指定前缀。
    """
    # 为每个键添加前缀，并返回新字典
    return {prefix + str(k): v for k, v in d.items()}

  

################################################ DSP Utils ################################################ 



def resample(
    waveforms: Tensor,
    factor_in: int,
    factor_out: int,
    rolloff: float = 0.99,
    lowpass_filter_width: int = 6,
) -> Tensor:
    """Resamples a waveform using sinc interpolation, adapted from torchaudio"""
    """
    使用 sinc 插值对音频波形进行重采样，适配自 torchaudio。

    参数:
    - waveforms (Tensor): 输入音频波形，形状为 (batch_size, channels, length)。
    - factor_in (int): 输入采样率因子。
    - factor_out (int): 输出采样率因子。
    - rolloff (float, 可选): 滤波器滚降因子，默认为0.99。
    - lowpass_filter_width (int, 可选): 低通滤波器宽度，默认为6。

    返回:
    - Tensor: 重采样后的音频波形。
    """
    # 获取批次大小、通道数和长度
    b, _, length = waveforms.shape
    # 计算目标长度
    length_target = int(factor_out * length / factor_in)
    # 获取设备和数据类型
    d = dict(device=waveforms.device, dtype=waveforms.dtype)

    # 计算基础因子
    base_factor = min(factor_in, factor_out) * rolloff
    # 计算滤波器宽度
    width = ceil(lowpass_filter_width * factor_in / base_factor)
    # 生成索引张量，范围从 -width 到 width + factor_in
    idx = torch.arange(-width, width + factor_in, **d)[None, None] / factor_in  # type: ignore # noqa
    # 生成时间步张量，范围从 0 到 -factor_out，步长为 -1
    t = torch.arange(0, -factor_out, step=-1, **d)[:, None, None] / factor_out + idx  # type: ignore # noqa
    # 将时间步张量缩放到 [-lowpass_filter_width, lowpass_filter_width] 并乘以 pi
    t = (t * base_factor).clamp(-lowpass_filter_width, lowpass_filter_width) * pi

    # 计算窗口函数
    window = torch.cos(t / lowpass_filter_width / 2) ** 2
    # 计算缩放因子
    scale = base_factor / factor_in
    # 计算 sinc 核函数，如果 t 为0，则设置为1.0
    kernels = torch.where(t == 0, torch.tensor(1.0).to(t), t.sin() / t)
    # 应用窗口函数和缩放因子
    kernels *= window * scale

    # 重塑波形张量形状为 (batch_size * channels, length)
    waveforms = rearrange(waveforms, "b c t -> (b c) t")
    # 对波形进行填充
    waveforms = F.pad(waveforms, (width, width + factor_in))
    # 使用一维卷积进行重采样，卷积核为 sinc 核，步幅为 factor_in
    resampled = F.conv1d(waveforms[:, None], kernels, stride=factor_in)
    # 重塑重采样后的波形张量形状为 (batch_size, channels, length_target)
    resampled = rearrange(resampled, "(b c) k l -> b c (l k)", b=b)
    # 返回目标长度的重采样波形
    return resampled[..., :length_target]


def downsample(waveforms: Tensor, factor: int, **kwargs) -> Tensor:
    """
    对音频波形进行下采样。

    参数:
    - waveforms (Tensor): 输入音频波形。
    - factor (int): 下采样因子。
    - **kwargs: 其他传递给 resample 的关键字参数。

    返回:
    - Tensor: 下采样后的音频波形。
    """
    # 调用 resample 进行下采样
    return resample(waveforms, factor_in=factor, factor_out=1, **kwargs)


def upsample(waveforms: Tensor, factor: int, **kwargs) -> Tensor:
    """
    对音频波形进行上采样。

    参数:
    - waveforms (Tensor): 输入音频波形。
    - factor (int): 上采样因子。
    - **kwargs: 其他传递给 resample 的关键字参数。

    返回:
    - Tensor: 上采样后的音频波形。
    """
    # 调用 resample 进行上采样
    return resample(waveforms, factor_in=1, factor_out=factor, **kwargs)



################################################ Torch Utils ################################################ 


def randn_like(tensor: Tensor, *args, generator: Optional[Generator] = None, **kwargs):
    """randn_like that supports generator"""
    """
    生成一个与输入张量形状相同的随机张量，支持使用随机数生成器。

    参数:
    - tensor (Tensor): 输入张量。
    - *args: 其他位置参数传递给 torch.randn。
    - generator (Optional[Generator], 可选): 随机数生成器，默认为 None。
    - **kwargs: 其他关键字参数传递给 torch.randn。

    返回:
    - Tensor: 与输入张量形状相同的随机张量。
    """
    # 生成随机张量并移动到与输入张量相同的设备
    return torch.randn(tensor.shape, *args, generator=generator, **kwargs).to(tensor)
