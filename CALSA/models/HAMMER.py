from functools import partial
from models.vit import VisionTransformer, interpolate_pos_embed
from models.xbert import BertConfig, BertForMaskedLM, BertForTokenClassification
import os
import torch
import torch.nn.functional as F
from torch import nn
import json
import numpy as np
import random

from models import box_ops
from tools.multilabel_metrics import get_multi_label
from timm.models.layers import trunc_normal_



class EnhancedPseudoLabelGenerator:
    """
    增强伪标签生成器 - 不依赖GT的多种策略
    """
    
    def __init__(self, confidence_threshold=0.5, noise_schedule='cosine'):
        self.confidence_threshold = confidence_threshold
        self.noise_schedule = noise_schedule
        self.epoch = 0
        self.history_predictions = []  # 存储历史预测用于多样性分析
        
    def set_epoch(self, epoch):
        self.epoch = epoch
        
    def get_noise_level(self, epoch, total_epochs):
        """动态噪声调度"""
        if self.noise_schedule == 'cosine':
            return 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))
        elif self.noise_schedule == 'linear':
            return max(0.1, 1.0 - epoch / total_epochs)
        else:
            return 0.3  # 固定噪声
    
    def uncertainty_based_enhancement(self, pred_coord, model_output=None):
        """
        基于模型不确定性的增强
        利用模型内部的不确定性信号
        """
        batch_size = pred_coord.size(0)
        
        # 方法1: 基于预测值的方差作为不确定性指标
        coord_variance = torch.var(pred_coord, dim=1, keepdim=True)
        uncertainty = coord_variance.mean(dim=1, keepdim=True)
        
        # 方法2: 如果有dropout，可以通过多次前向传播估计不确定性
        # 这里模拟不确定性（实际中可以通过Monte Carlo Dropout实现）
        simulated_uncertainty = torch.rand(batch_size, 1, device=pred_coord.device)
        
        # 结合两种不确定性
        total_uncertainty = (uncertainty + simulated_uncertainty) / 2
        
        # 不确定性越高，添加的噪声越大
        noise_scale = total_uncertainty * 0.2  # 调节噪声强度
        noise = torch.randn_like(pred_coord) * noise_scale
        
        enhanced_coord = pred_coord + noise
        return torch.clamp(enhanced_coord, 0, 1)  # 确保坐标在有效范围
    
    def ensemble_diversity_enhancement(self, pred_coord):
        """
        模拟集成学习的多样性
        通过人工创建多个"视角"来增加多样性
        """
        batch_size = pred_coord.size(0)
        enhanced_coords = []
        
        # 创建多个扰动版本
        for i in range(3):  # 创建3个版本
            # 不同的扰动策略
            if i == 0:
                # 位置偏移
                offset = torch.randn(batch_size, 2, device=pred_coord.device) * 0.05
                coord_copy = pred_coord.clone()
                coord_copy[:, :2] += offset  # 只对cx, cy添加偏移
                
            elif i == 1:
                # 尺寸缩放
                scale = torch.rand(batch_size, 2, device=pred_coord.device) * 0.4 + 0.8  # 0.8-1.2倍缩放
                coord_copy = pred_coord.clone()
                coord_copy[:, 2:] *= scale  # 只对w, h进行缩放
                
            else:
                # 组合扰动
                noise = torch.randn_like(pred_coord) * 0.1
                coord_copy = pred_coord + noise
                
            enhanced_coords.append(coord_copy)
        
        # 随机选择其中一个版本
        selected_idx = torch.randint(0, 3, (batch_size,))
        final_coord = torch.stack(enhanced_coords)[selected_idx, torch.arange(batch_size)]
        
        return torch.clamp(final_coord, 0, 1)
    
    def confidence_aware_augmentation(self, pred_coord, confidence_scores=None):
        """
        基于置信度的自适应增强
        置信度低的预测添加更多噪声
        """
        batch_size = pred_coord.size(0)
        
        if confidence_scores is None:
            # 简单的置信度估计：坐标值离边界越远置信度越高
            center_distance = torch.abs(pred_coord[:, :2] - 0.5).mean(dim=1)
            size_confidence = torch.min(pred_coord[:, 2:], dim=1)[0]
            confidence_scores = (1 - center_distance) * size_confidence
        
        # 置信度越低，噪声越大
        noise_scale = (1 - confidence_scores).unsqueeze(1) * 0.3
        noise = torch.randn_like(pred_coord) * noise_scale
        
        enhanced_coord = pred_coord + noise
        return torch.clamp(enhanced_coord, 0, 1)
    
    def temporal_consistency_enhancement(self, pred_coord):
        """
        时间一致性增强
        基于历史预测的变化来调整当前预测
        """
        self.history_predictions.append(pred_coord.clone().detach())
        
        # 保持最近5次预测
        if len(self.history_predictions) > 5:
            self.history_predictions.pop(0)
        
        if len(self.history_predictions) < 2:
            return pred_coord
        
        # 计算预测的变化趋势
        recent_change = self.history_predictions[-1] - self.history_predictions[-2]
        
        # 如果变化太大，添加稳定性噪声
        change_magnitude = torch.norm(recent_change, dim=1, keepdim=True)
        stability_noise = torch.randn_like(pred_coord) * change_magnitude * 0.1
        
        enhanced_coord = pred_coord + stability_noise
        return torch.clamp(enhanced_coord, 0, 1)
    
    def curriculum_based_enhancement(self, pred_coord, epoch, total_epochs):
        """
        课程学习增强
        训练初期添加更多噪声，后期逐渐减少
        """
        # 动态噪声水平
        noise_level = self.get_noise_level(epoch, total_epochs)
        
        # 早期训练：更强的增强
        if epoch < total_epochs * 0.3:
            # 强增强策略
            enhanced_coord = self.ensemble_diversity_enhancement(pred_coord)
            extra_noise = torch.randn_like(pred_coord) * noise_level
            enhanced_coord = enhanced_coord + extra_noise
            
        elif epoch < total_epochs * 0.7:
            # 中等增强策略
            enhanced_coord = self.confidence_aware_augmentation(pred_coord)
            
        else:
            # 轻微增强策略
            enhanced_coord = self.uncertainty_based_enhancement(pred_coord)
        
        return torch.clamp(enhanced_coord, 0, 1)
    
    def multi_scale_enhancement(self, pred_coord):
        """
        多尺度增强
        在不同尺度上添加不同类型的噪声
        """
        batch_size = pred_coord.size(0)
        
        # 细粒度噪声（高频）
        fine_noise = torch.randn_like(pred_coord) * 0.02
        
        # 中等粒度噪声（中频）
        medium_noise = torch.randn_like(pred_coord) * 0.1
        
        # 粗粒度噪声（低频）
        coarse_noise = torch.randn_like(pred_coord) * 0.3
        
        # 随机选择噪声类型
        noise_type = torch.randint(0, 3, (batch_size,))
        final_noise = torch.zeros_like(pred_coord)
        
        fine_mask = (noise_type == 0).unsqueeze(1)
        medium_mask = (noise_type == 1).unsqueeze(1)
        coarse_mask = (noise_type == 2).unsqueeze(1)
        
        final_noise = (fine_mask * fine_noise + 
                      medium_mask * medium_noise + 
                      coarse_mask * coarse_noise)
        
        enhanced_coord = pred_coord + final_noise
        return torch.clamp(enhanced_coord, 0, 1)

def apply_enhanced_pseudo_labeling(model_pred_coord, epoch, total_epochs, strategy='curriculum'):
    """
    应用增强伪标签策略的主函数
    """
    enhancer = EnhancedPseudoLabelGenerator()
    enhancer.set_epoch(epoch)
    
    if strategy == 'curriculum':
        return enhancer.curriculum_based_enhancement(model_pred_coord, epoch, total_epochs)
    elif strategy == 'uncertainty':
        return enhancer.uncertainty_based_enhancement(model_pred_coord)
    elif strategy == 'ensemble':
        return enhancer.ensemble_diversity_enhancement(model_pred_coord)
    elif strategy == 'confidence':
        return enhancer.confidence_aware_augmentation(model_pred_coord)
    elif strategy == 'multi_scale':
        return enhancer.multi_scale_enhancement(model_pred_coord)
    elif strategy == 'temporal':
        return enhancer.temporal_consistency_enhancement(model_pred_coord)
    else:
        # 组合策略：随机选择
        strategies = ['uncertainty', 'ensemble', 'confidence', 'multi_scale']
        selected = random.choice(strategies)
        return apply_enhanced_pseudo_labeling(model_pred_coord, epoch, total_epochs, selected)

# 在你的模型中的应用示例
def integrate_enhanced_pseudo_labels(self, output_coord, epoch, total_epochs=20):
    """
    集成到你的HAMMER模型中
    """
    with torch.no_grad():
        # 使用增强的伪标签策略
        enhanced_coord = apply_enhanced_pseudo_labeling(
            output_coord, epoch, total_epochs, strategy='curriculum'
        )
        
        # 生成fake_map
        cx, cy, w, h = enhanced_coord[:, 0], enhanced_coord[:, 1], enhanced_coord[:, 2], enhanced_coord[:, 3]
        fake_map = generate_fakemap(cx, cy, w, h, image_size=(256, 256))
        
    return fake_map, enhanced_coord


def save_coordinates_to_txt(cx, cy, w, h, filename="coordinates.txt"):
    """
    将原始坐标和带噪声的坐标保存到文本文件
    
    参数:
        cx, cy, w, h: 原始坐标 (torch.Tensor 或 numpy.ndarray)
        cx_n, cy_n, w_n, h_n: 带噪声坐标 (torch.Tensor 或 numpy.ndarray)
        filename: 输出文件名 (默认: coordinates.txt)
    """
    # 确保是numpy数组格式 (如果是torch.Tensor则转换)
    if hasattr(cx, 'cpu'):  # 检查是否是torch.Tensor
        cx = cx.cpu().numpy()
        cy = cy.cpu().numpy()
        w = w.cpu().numpy()
        h = h.cpu().numpy()
       
    
    # 格式化数据为字符串
    data = []
    headers = ["Index", "cx", "cy", "w", "h"]
    data.append("\t".join(headers))
    
    for i in range(len(cx)):
        row = [
            str(i),
            f"{cx[i]:.6f}",
            f"{cy[i]:.6f}",
            f"{w[i]:.6f}",
            f"{h[i]:.6f}",
          
        ]
        data.append("\t".join(row))
    
    # 写入文件
    with open(filename, 'w') as f:
        f.write("\n".join(data))
    
    print(f"坐标已保存到 {filename}")

def KL_divergence(p, q, epsilon=1e-8):
        q = q + epsilon
        kl_div = p * torch.log(p / q)
        return kl_div.sum()

def L_i2t(V, T):
    p = F.softmax(V, dim=-1)
    q = F.softmax(T, dim=-1)
    return KL_divergence(p, q)

def L_t2i(V, T):
    p = F.softmax(T, dim=-1)
    q = F.softmax(V, dim=-1)
    return KL_divergence(p, q)

def L_cmpm(V, T):
    L_i2t_loss = L_i2t(V, T)
    L_t2i_loss = L_t2i(V, T)
    return L_i2t_loss + L_t2i_loss
def generate_patch_labels(images, norm_bboxes, patch_size=16):
    """
    修正维度广播问题的版本
    """
    B, C, H, W = images.shape
    device = images.device
    
    # 确保输入为正方形且可被分块
    assert H == W and H % patch_size == 0, f"需要正方形输入且可被{patch_size}整除"
    num_patches = H // patch_size
    
    # 转换归一化坐标到绝对坐标
    cx = norm_bboxes[..., 0] * W
    cy = norm_bboxes[..., 1] * H
    w = norm_bboxes[..., 2] * W
    h = norm_bboxes[..., 3] * H
    
    xmin = torch.clamp(cx - w/2, 0, W)
    ymin = torch.clamp(cy - h/2, 0, H)
    xmax = torch.clamp(cx + w/2, 0, W)
    ymax = torch.clamp(cy + h/2, 0, H)
    
    # 生成块网格 [P,P,4]
    grid = torch.stack(torch.meshgrid(
        torch.arange(num_patches, device=device) * patch_size,
        torch.arange(num_patches, device=device) * patch_size,
    ), dim=-1)
    blocks = torch.cat([
        grid, 
        grid + patch_size
    ], dim=-1)  # [P,P,4] (x1,y1,x2,y2)
    
    # 调整维度用于广播 [B,N,1,1,4] vs [1,1,P,P,4]
    boxes = torch.stack([xmin, ymin, xmax, ymax], dim=-1).to(device)  # [B,N,4]
    boxes = boxes.view(B, -1, 1, 1, 4)  # 关键修正点
    blocks = blocks.view(1, 1, num_patches, num_patches, 4)
    
    # 计算交集区域
    inter_x1 = torch.maximum(blocks[..., 0], boxes[..., 0])
    inter_y1 = torch.maximum(blocks[..., 1], boxes[..., 1])
    inter_x2 = torch.minimum(blocks[..., 2], boxes[..., 2])
    inter_y2 = torch.minimum(blocks[..., 3], boxes[..., 3])
    
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    patch_labels = (inter_area > 0).any(dim=1).view(B, -1).float()
    
    return patch_labels


def generate_fakemap(cx, cy, w, h, image_size=256):
    """
    生成伪造区域的高斯热图（fake map）
    
    参数:
        cx (float): 伪造区域的中心x坐标
        cy (float): 伪造区域的中心y坐标
        w (float): 伪造区域的宽度
        h (float): 伪造区域的高度
        image_size (int): 输入图像的尺寸，默认为256
    
    返回:
        np.ndarray: 256x256的fake map矩阵，值范围[0, 1]
    """
       # 统一处理图像尺寸输入
    if isinstance(image_size, int):
        H, W = image_size, image_size
    else:
        H, W = image_size  # 拆分为高度和宽度
    device = cx.device
    original_size=(256, 256)
    W_orig, H_orig = original_size  # 原始图像尺寸
    
    # === 关键步骤1：反归一化坐标和尺寸 ===
    cx_pixel = cx * W_orig  # [B] 原始像素坐标
    cy_pixel = cy * H_orig  # [B]
    w_pixel = w * W_orig    # [B]
    h_pixel = h * H_orig    # [B]
    
    # === 关键步骤2：计算sigma（基于原始尺寸）===
    sigma = torch.sqrt((h_pixel/2)**2 + (w_pixel/2)**2)  # [B]
    
    # === 生成坐标网格 ===
    x = torch.arange(image_size[0], device=device).float()  # [image_size]
    y = torch.arange(image_size[1], device=device).float()
    yy, xx = torch.meshgrid(y, x, indexing='ij')  # [image_size, image_size]
    
    # === 扩展维度以支持广播 ===
    xx = xx.unsqueeze(0)  # [1, H, W]
    yy = yy.unsqueeze(0)
    cx_pixel = cx_pixel.view(-1, 1, 1)  # [B, 1, 1]
    cy_pixel = cy_pixel.view(-1, 1, 1)
    sigma = sigma.view(-1, 1, 1)
    
    # === 计算高斯分布 ===
    distance_sq = (xx - cx_pixel)**2 + (yy - cy_pixel)**2  # [B, H, W]
    fake_map = torch.exp(-distance_sq / (2 * (sigma**2) + 1e-8))

    # 限制值在0到1之间
    fake_map = torch.clamp(fake_map, 0, 1)
    return fake_map



def generate_noisy_coordinates(cx, cy, w, h, noise_level=1):
    """
    为批次中的每个边界框坐标添加随机噪声，模拟模型预测的效果，并保持大约85%的IoU重叠。
    
    参数:
        cx (Tensor): GT边界框的中心x坐标，形状为 [batchsize]
        cy (Tensor): GT边界框的中心y坐标，形状为 [batchsize]
        w (Tensor): GT边界框的宽度，形状为 [batchsize]
        h (Tensor): GT边界框的高度，形状为 [batchsize]
        noise_level (float): 控制噪声的大小（默认为0.1）
        
    返回:
        Tuple: 添加噪声后的坐标 (cx', cy', w', h')，应具有 ~85% 的IoU重叠
    """
    # 对每个边界框的坐标添加噪声
    cx_noise = torch.randn_like(cx) * noise_level * w  # 随机噪声与宽度成比例
    cy_noise = torch.randn_like(cy) * noise_level * h  # 随机噪声与高度成比例
    w_noise = torch.randn_like(w) * noise_level * w  # 随机噪声对于宽度
    h_noise = torch.randn_like(h) * noise_level * h  # 随机噪声对于高度
    
    # 将噪声添加到GT坐标
    noisy_cx = cx + cx_noise
    noisy_cy = cy + cy_noise
    noisy_w = w + w_noise
    noisy_h = h + h_noise

    # 选择全为0的样本并修改其坐标为随机值
    zero_idx = (cx == 0) & (cy == 0) & (w == 0) & (h == 0)  # 找到全为0的样本
    if zero_idx.any():
        random_zero_idx = torch.nonzero(zero_idx).squeeze()  # 获取全为0样本的索引
        if len(random_zero_idx) > 0:
            selected_zero_idx = random_zero_idx[torch.randint(len(random_zero_idx), (4,))]
            # 将该样本的坐标修改为在 [0, 1] 范围内的随机值
            noisy_cx[selected_zero_idx] = torch.rand(1).item() * 0.5 + 1e-2   # 随机生成 [0, 1] 范围内的值
            noisy_cx[selected_zero_idx] = torch.rand(1).item() * 0.5 + 1e-2 
            noisy_cx[selected_zero_idx] = torch.rand(1).item() * 0.5 + 1e-2 
            noisy_cx[selected_zero_idx] = torch.rand(1).item() * 0.5 + 1e-2 

    # 选择非全0的样本并修改其坐标为 [0, 0, 0, 0]
    non_zero_idx = (cx != 0) | (cy != 0) | (w != 0) | (h != 0)  # 找到非全0的样本
    if non_zero_idx.any():
        random_non_zero_idx = torch.nonzero(non_zero_idx).squeeze()  # 获取非全0样本的索引
        if len(random_non_zero_idx) > 1:
            selected_non_zero_idx = random_non_zero_idx[torch.randint(len(random_non_zero_idx), (6,))]  # 随机选择两个样本
            # 将这些样本的坐标修改为 [0, 0, 0, 0]
            noisy_cx[selected_non_zero_idx] = 0.0
            noisy_cy[selected_non_zero_idx] = 0.0
            noisy_w[selected_non_zero_idx] = 0.0
            noisy_h[selected_non_zero_idx] = 0.0
    
    return noisy_cx, noisy_cy, noisy_w, noisy_h

class DIMD(nn.Module):
    def __init__(self, num_queries=32, hidden_dim=768, num_layers=6):
        super(DIMD, self).__init__()
        
        # 初始化查询（篡改查询和内容查询）
        self.num_queries = num_queries
        self.query_dim = hidden_dim
        
        # 随机初始化查询
        self.Qf = nn.Parameter(torch.zeros(1, 1, hidden_dim))  # 篡改查询
        self.Qc = nn.Parameter(torch.zeros(1, 1, hidden_dim))  # 内容查询
        
        # 位置编码（对 Qf 和 Qc 都使用相同的编码）
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        # 交叉注意力模块（与篡改和内容特征交互）
        self.cross_attention_f = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=12, dropout=0.0, batch_first=True)
        self.cross_attention_c = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=12, dropout=0.0, batch_first=True)
        
        # 自注意力模块（用于在两个分支间传播信息）
        self.self_attention_f = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=12, dropout=0.0, batch_first=True)
        self.self_attention_c = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=12, dropout=0.0, batch_first=True)
        
        # 前馈神经网络（FFN）
        self.ffn_f = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.ffn_c = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        trunc_normal_(self.Qf, std=.02)
        trunc_normal_(self.Qc, std=.02)
        trunc_normal_(self.positional_encoding, std=.02)


    def forward(self, Vf, Vc):
        bs = Vf.size(0)
        Ql_f = self.Qf.expand(bs, -1, -1)  # 扩展篡改查询
        Ql_c = self.Qc.expand(bs, -1, -1)  # 扩展内容查询
        # 对查询应用位置编码
        Ql_f = Ql_f + self.positional_encoding
        Ql_c = Ql_c + self.positional_encoding
        
        # 与篡改和内容特征应用交叉注意力
        Ql_f, _ = self.cross_attention_f(Ql_f, Vf, Vf)
        Ql_c, _ = self.cross_attention_c(Ql_c, Vc, Vc)
        
        # 在合并查询时，分离查询的梯度
        Ql_f_detached = Ql_f.detach()
        Ql_c_detached = Ql_c.detach()
        
        # 合并篡改查询和内容查询
        Ql_star_f = torch.cat((Ql_f, Ql_c_detached), dim=1) #[16,2,768]
        Ql_star_c = torch.cat((Ql_f_detached, Ql_c), dim=1) #[16,2,768]
        
        # 应用共享的自注意力传播信息
        Ql_star_f, _ = self.self_attention_f(Ql_star_f, Ql_star_f, Ql_star_f)
        Ql_star_c, _ = self.self_attention_c(Ql_star_c, Ql_star_c, Ql_star_c)
        
        # 通过前馈神经网络（FFN）处理最终的查询表示
        Ql_plus_f = self.ffn_f(Ql_star_f.squeeze(1)) #[16,2,768]
        Ql_plus_c = self.ffn_c(Ql_star_c.squeeze(1))
        
        # 返回篡改和内容表示
        return Ql_plus_f, Ql_plus_c




class PSILLoss(nn.Module):
    def __init__(self, temp=0.07):
        super().__init__()
        self.temp = temp
        
    def forward(self, Vf, patch_labels):
        """
        改进版PSIL损失，增加温度系数
        """ 
        # 去除class token
        Vf = Vf[:, 1:, :]  # [B,N,D]
        
        # 特征归一化
        Vf_norm = F.normalize(Vf, p=2, dim=-1)
        
        # 计算相似度矩阵
        sim_matrix = torch.bmm(Vf_norm, Vf_norm.transpose(1,2)) / self.temp  # [B,N,N]
        
        # 生成目标矩阵
        target = (patch_labels.unsqueeze(2) == patch_labels.unsqueeze(1)).float()
        
        # 计算加权交叉熵
        loss = F.binary_cross_entropy_with_logits(
            sim_matrix, 
            target,
            reduction='none'
        )
        
        # 屏蔽对角线
        mask = torch.eye(sim_matrix.size(1), dtype=torch.bool, device=sim_matrix.device)
        loss = loss.masked_fill(mask, 0).mean()
        
        return loss

class CSRA_Transformer(nn.Module):
    def __init__(self, num_classes, feature_dim, lambda_init=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.conv_attention = nn.ModuleList([
            nn.Conv2d(feature_dim, 1, kernel_size=1) for _ in range(num_classes)
        ])
        self.fc_global = nn.Linear(feature_dim, num_classes)  # Class Token分类层
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init))
        
    def forward(self, patch_tokens, class_token):
        # patch_tokens: (B, N^2, D)
        # class_token: (B, D)
        
        # 重塑为空间特征 (B, H, W, D)
        # print('patch_tokens:',patch_tokens.shape)
        # print('class_token:',class_token.shape)
        B, seq_len, D = patch_tokens.shape
        h = w = int(seq_len ** 0.5)
        spatial_features = patch_tokens.view(B, h, w, D).permute(0, 3, 1, 2)  # (B, D, H, W)
        
        # 全局得分
        S_global = self.fc_global(class_token)  # (B, C)
        
        # 计算每个类别的注意力得分
        S_attn = []
        for c in range(self.num_classes):
            attn_map = torch.sigmoid(self.conv_attention[c](spatial_features))  # (B, 1, H, W)
            weighted_feature = attn_map * spatial_features  # (B, D, H, W)
            pooled = torch.mean(weighted_feature, dim=(2,3))  # (B, D)
            S_c = torch.mean(pooled, dim=1)  # (B,) 或其他聚合方式
            S_attn.append(S_c)
        
        S_attn = torch.stack(S_attn, dim=1)  # (B, C)
        
        # 残差融合
        S_final = S_global + self.lambda_param * S_attn
        return S_final

class Ep(nn.Module):
    def __init__(self, input_size=256, output_dim=768):
        super().__init__()
        # 假设输入fake_map为 [B,1,256,256]
        self.conv_net = nn.Sequential(
            # 第1层卷积: 下采样到 [B,64,128,128]
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # 第2层卷积: 下采样到 [B,128,64,64]
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # 第3层卷积: 下采样到 [B,256,32,32]
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # 自适应池化到固定尺寸 [B,256,16,16]
            nn.AdaptiveAvgPool2d((16,16)),
            
            # 1x1卷积调整通道数到768
            nn.Conv2d(256, 768, kernel_size=1),
        )
        
    def forward(self, fake_map):
        # 输入 fake_map: [B,1,H,W]
        x = self.conv_net(fake_map)  # 输出 [B,768,16,16]
        x = x.flatten(2).permute(0,2,1)  # 转换为 [B,256,768]
        return x

 

class HAMMER(nn.Module):
    def __init__(self, 
                 args = None, 
                 config = None,               
                 text_encoder = None,
                 tokenizer = None,
                 init_deit = True\
                 ):
        super().__init__()
        
        self.batch_count = 0
        self.args = args
        self.tokenizer = tokenizer 
        embed_dim = config['embed_dim']
     
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))   
        
        # self.visual_encoder_f = VisionTransformer(
        #     img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
        #     mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        
        if init_deit:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
            state_dict['pos_embed'] = pos_embed_reshaped
            msg = self.visual_encoder.load_state_dict(state_dict,strict=False)
            print(msg)          
        vision_width = config['vision_width']       
        bert_config = BertConfig.from_json_file(config['bert_config'])
        bert_config_f = BertConfig.from_json_file(config['bert_config_f'])
        self.text_encoder_f_1 = BertForTokenClassification.from_pretrained(args.text_encoder, 
                                                                    config=bert_config_f, 
                                                                    label_smoothing=config['label_smoothing']) 
        self.text_encoder = BertForTokenClassification.from_pretrained(args.text_encoder, 
                                                                    config=bert_config, 
                                                                    label_smoothing=config['label_smoothing'])      

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)         

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])   
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']  

        # creat itm head
        self.itm_head = self.build_mlp(input_dim=text_width, output_dim=2)

        # creat bbox head
        self.bbox_head = self.build_mlp(input_dim=text_width, output_dim=4)
        self.bbox_head_f = self.build_mlp(input_dim=text_width, output_dim=4)

        self.bbox_head_c = self.build_mlp(input_dim=text_width, output_dim=4)


        # creat multi-cls head
        self.cls_head = self.build_mlp(input_dim=text_width, output_dim=2)
        self.cls_head_A = self.build_mlp(input_dim=text_width, output_dim=4)
        self.CARA = CSRA_Transformer(num_classes=2, feature_dim=text_width, lambda_init=0.3)
        self.psil = PSILLoss()
        self.dimd = DIMD()
        self.MIG = Ep(input_size=(256, 256), output_dim=embed_dim)  # fake map encoder

        # create momentum models
        self.visual_encoder_m = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)) 
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertForTokenClassification.from_pretrained(args.text_encoder, 
                                                                    config=bert_config,
                                                                    label_smoothing=config['label_smoothing'])       
        self.text_proj_m = nn.Linear(text_width, embed_dim)    
        
        self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                            [self.vision_proj,self.vision_proj_m],
                            [self.text_encoder,self.text_encoder_m],
                            [self.text_proj,self.text_proj_m],
                           ]
        
        self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  
                             
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        self.norm_layer_aggr =nn.LayerNorm(text_width)
        self.cls_token_local = nn.Parameter(torch.zeros(1, 1, text_width))
        self.cls_token_local_e = nn.Parameter(torch.zeros(1, 1, text_width))

        self.aggregator = nn.MultiheadAttention(text_width, 12, dropout=0.0, batch_first=True)

        self.norm_layer_it_cross_atten =nn.LayerNorm(text_width)
        self.it_cross_attn = nn.MultiheadAttention(text_width, 12, dropout=0.0, batch_first=True)
        trunc_normal_(self.cls_token_local, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def build_mlp(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim* 2, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, output_dim)
        )


    def get_bbox_loss(self, output_coord, target_bbox_ex, is_image=None, target_bbox_map_ids=None):
        """
        Bounding Box Loss: L1 & GIoU

        Args:
            image_embeds: encoding full images
        """
        n_objs = output_coord.size(0)
        n_bbox = target_bbox_ex.size(0)

        assert n_objs == n_bbox
        target_bbox = target_bbox_ex

        loss_bbox = F.l1_loss(output_coord, target_bbox, reduction='none')  # bsz, 4
        boxes1 = box_ops.box_cxcywh_to_xyxy(output_coord)
        boxes2 = box_ops.box_cxcywh_to_xyxy(target_bbox)
        if (boxes1[:, 2:] < boxes1[:, :2]).any() or (boxes2[:, 2:] < boxes2[:, :2]).any():
            # early check of degenerated boxes
            print("### (boxes1[:, 2:] < boxes1[:, :2]).any() or (boxes2[:, 2:] < boxes2[:, :2]).any()")
            loss_giou = torch.zeros(output_coord.size(0), device=output_coord.device)
        else:
            loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(boxes1, boxes2))  # bsz

        if is_image is None:
            num_boxes = target_bbox.size(0)
        else:
            num_boxes = torch.sum(1 - is_image)
            loss_bbox = loss_bbox * (1 - is_image.view(-1, 1))
            loss_giou = loss_giou * (1 - is_image)

        return loss_bbox.sum() / num_boxes, loss_giou.sum() / num_boxes
    
    def forward(self, image, label, text, fake_image_box, fake_text_pos, epoch,  alpha=0,is_train=True):
        
            if is_train:
                pass
            else:
                image_embeds, image_embeds_b = self.visual_encoder(image) 
                image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

                text_output = self.text_encoder.bert(text.input_ids, attention_mask = text.attention_mask,                      
                                                return_dict = True, mode = 'text')            
                text_embeds = text_output.last_hidden_state

                
                ##================= IMG ========================## 
                bs = image.size(0)
                cls_tokens_local = self.cls_token_local.expand(bs, -1, -1)

                text_attention_mask_clone = text.attention_mask.clone() # [:,1:] for ingoring class token
                local_feat_padding_mask_text = text_attention_mask_clone==0 # 0 = pad token
                local_feat_it_cross_attn = image_embeds_b + self.it_cross_attn(query=self.norm_layer_it_cross_atten(image_embeds_b), 
                                                key=self.norm_layer_it_cross_atten(text_embeds), 
                                                value=self.norm_layer_it_cross_atten(text_embeds),
                                                key_padding_mask=local_feat_padding_mask_text)[0]

                local_feat_aggr = self.aggregator(query=self.norm_layer_aggr(cls_tokens_local), 
                                                key=self.norm_layer_aggr(local_feat_it_cross_attn[:,1:,:]), 
                                                value=self.norm_layer_aggr(local_feat_it_cross_attn[:,1:,:]))[0]
                output_coord = self.bbox_head(local_feat_aggr.squeeze(1)).sigmoid()
                cx_p, cy_p, w_p, h_p = output_coord[:,0], output_coord[:,1], output_coord[:,2], output_coord[:,3]

                fake_map = generate_fakemap(cx_p, cy_p, w_p, h_p,image_size=(256,256))

                fake_map = torch.tensor(fake_map).unsqueeze(1).to(image.device)  

                A = self.MIG(fake_map)
                cls_token = image_embeds_b[:, 0:1, :] 
                patch_features = image_embeds_b[:, 1:, :]  
                enhanced_patches = patch_features + A 

                enhanced_Vc = torch.cat([cls_token, enhanced_patches], dim=1)  

                output_pos = self.text_encoder.bert(encoder_embeds = text_embeds, 
                                                attention_mask = text.attention_mask,
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,      
                                                return_dict = True,
                                                mode = 'fusion',
                                            )               
                ##================= BIC ========================## 
                logits_real_fake = self.itm_head(output_pos.last_hidden_state[:,0,:])

                ##================= IMG_e ========================## 
                cls_tokens_local_e = self.cls_token_local_e.expand(bs, -1, -1)

                text_attention_mask_clone_e = text.attention_mask.clone() # [:,1:] for ingoring class token
                local_feat_padding_mask_text_e = text_attention_mask_clone_e==0 # 0 = pad token

                local_feat_it_cross_attn_e = enhanced_Vc + self.it_cross_attn(query=self.norm_layer_it_cross_atten(enhanced_Vc), 
                                                key=self.norm_layer_it_cross_atten(text_embeds), 
                                                value=self.norm_layer_it_cross_atten(text_embeds),
                                                key_padding_mask=local_feat_padding_mask_text_e)[0]

                local_feat_aggr_e = self.aggregator(query=self.norm_layer_aggr(cls_tokens_local_e), 
                                                key=self.norm_layer_aggr(local_feat_it_cross_attn_e[:,1:,:]), 
                                                value=self.norm_layer_aggr(local_feat_it_cross_attn_e[:,1:,:]))[0]
                output_coord_e = self.bbox_head(local_feat_aggr_e.squeeze(1)).sigmoid()
                
                # ##================= MLC ========================## 
                cross_embeds_cls = local_feat_it_cross_attn
                cls_f = self.CARA(cross_embeds_cls[:, 1:, :], cross_embeds_cls[:, 0, :])
                cls_t = self.cls_head(output_pos.last_hidden_state[:,0,:])

                logits_multicls = torch.concat((cls_f, cls_t), dim=1)
                ##================= TMG ========================##   
                input_ids = text.input_ids.clone()
                logits_tok = self.text_encoder(input_ids, 
                                            attention_mask = text.attention_mask,
                                            encoder_hidden_states = image_embeds,
                                            encoder_attention_mask = image_atts,      
                                            return_dict = True,
                                            return_logits = True,   
                                            )     
                return logits_real_fake, logits_multicls, output_coord_e, logits_tok
        
  

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
            
            
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        # print(f"queue_size: {self.queue_size}, batch_size: {batch_size}")
        # assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr 
        
        
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

