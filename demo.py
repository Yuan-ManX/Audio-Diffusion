import torch
import torch.nn.functional as F
from models import DiffusionAE, VDiffusion, VSampler
from components import LTPlugin, MelSpectrogram, UNetV0, XUNet


# 实例化 DiffusionAE（扩散自编码器）
autoencoder = DiffusionAE(
    # 使用 MelE1d 作为编码器，这是一种梅尔频谱图编码器
    encoder=MelSpectrogram( # The encoder used, in this case a mel-spectrogram encoder
        in_channels=2, # 输入通道数，2 表示立体声（左右声道）
        channels=512, # 编码器中间层的通道数
        multipliers=[1, 1], # 通道数乘法因子列表，用于调整每个层的通道数
        factors=[2], # 下采样因子列表，用于调整每个层的空间分辨率
        num_blocks=[12], # 每个下采样阶段的块数
        out_channels=32, # 编码器输出的通道数
        mel_channels=80, # 梅尔频谱图的通道数
        mel_sample_rate=48000, # 梅尔频谱图的采样率
        mel_normalize_log=True, # 是否对梅尔频谱图进行对数归一化
    ),
    inject_depth=6, # 注入编码器输出的深度（层数）
    net_t=UNetV0, # 使用 UNetV0 作为扩散模型的网络类型
    in_channels=2, # U-Net 的输入和输出通道数，2 表示立体声（左右声道）
    channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024], # U-Net 每个层的通道数序列
    factors=[1, 4, 4, 4, 2, 2, 2, 2, 2], # U-Net 每个层的下采样和上采样因子序列
    items=[1, 2, 2, 2, 2, 2, 2, 4, 4], # U-Net 每个层的重复次数序列
    diffusion_t=VDiffusion,  # 使用 VDiffusion 作为扩散方法
    sampler_t=VSampler, # 使用 VSampler 作为扩散采样器
)

# Train autoencoder with audio samples
# 使用音频样本训练自编码器
# 生成随机音频样本，形状为 [batch_size, in_channels, length]
audio = torch.randn(1, 2, 2**18) # [batch, in_channels, length]

# 前向传播，计算损失
loss = autoencoder(audio)

# 反向传播，更新模型参数
loss.backward()

# Encode/decode audio
# 编码和解码音频
# 生成随机音频样本，形状为 [batch_size, in_channels, length]
audio = torch.randn(1, 2, 2**18) # [batch, in_channels, length]

# 对音频进行编码，得到潜在空间表示
latent = autoencoder.encode(audio) # Encode

# 使用扩散模型对潜在空间进行解码，num_steps 表示采样步骤数量
# 解码并生成音频样本
sample = autoencoder.decode(latent, num_steps=10) # Decode by sampling diffusion model conditioning on latent
