# -*- coding: utf-8 -*-
"""
BALM_MedualTime 主运行脚本
融合BALM-TSF文本描述和MedualTime双适配器架构
"""

import argparse
import importlib.util
import inspect
import os
import random
import time
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"Pandas requires version '1\.3\.6' or newer of 'bottleneck'.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"`resume_download` is deprecated.*",
    category=FutureWarning,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import numpy as np
from torch import optim
from torch.optim import lr_scheduler

from data_provider.data_factory import data_provider
from utils.logger import get_logger
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

# 设置环境变量
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"huggingface_hub\.file_download",
)
warnings.filterwarnings(
    "ignore",
    message=r"`resume_download` is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Pandas requires version '1\.3\.6' or newer of 'bottleneck'.*",
    category=UserWarning,
)


def get_args():
    """获取命令行参数"""
    parser = argparse.ArgumentParser(description="BALM_MedualTime")

    # 基本配置
    parser.add_argument("--is_training", type=int, default=1, help="是否训练")
    parser.add_argument("--model_id", type=str, default="test", help="模型ID")
    parser.add_argument("--model", type=str, default="BALM_MedualTime", help="模型名称")
    parser.add_argument("--model_file", type=str, default="", help="可选：从指定模型文件加载 BALM_MedualTime 类")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    # 数据配置
    parser.add_argument("--data", type=str, default="ETTh2", help="数据集类型")
    parser.add_argument("--root_path", type=str, default="./dataset/ETT-small/", help="数据根目录")
    parser.add_argument("--data_path", type=str, default="ETTh2.csv", help="数据文件名")
    parser.add_argument("--features", type=str, default="M", 
                        help="预测任务: M-多变量预测多变量, S-单变量预测单变量, MS-多变量预测单变量")
    parser.add_argument("--target", type=str, default="OT", help="S或MS任务中的目标特征")
    parser.add_argument("--freq", type=str, default="h", help="时间特征编码频率")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/", help="模型检查点保存位置")

    # 预测任务配置
    parser.add_argument("--seq_len", type=int, default=512, help="输入序列长度")
    parser.add_argument("--label_len", type=int, default=48, help="起始标记长度")
    parser.add_argument("--pred_len", type=int, default=96, help="预测序列长度")
    parser.add_argument("--seasonal_patterns", type=str, default="Monthly", help="M4数据集的子集")

    # 模型配置
    parser.add_argument("--enc_in", type=int, default=7, help="编码器输入大小（变量数）")
    parser.add_argument("--d_model", type=int, default=768, help="模型维度")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout比例")
    parser.add_argument("--embed", type=str, default="timeF", help="时间特征编码方式")
    parser.add_argument("--semantic_freq_topk", type=int, default=4, help="每个patch保留的显著频率分量数")
    parser.add_argument("--semantic_hidden_dim", type=int, default=384, help="神经符号描述符投影MLP隐层维度")
    parser.add_argument("--decomp_kernel", type=int, default=25, help="预测头序列分解的平滑核大小")
    parser.add_argument("--router_hidden_dim", type=int, default=384, help="逐步专家路由器隐层维度")
    parser.add_argument(
        "--spectral_scales",
        type=str,
        default="0.5,1.0,2.0",
        help="多尺度谱域适配器比例，逗号分隔",
    )
    
    # 适配器配置
    parser.add_argument("--adapter_layers", type=int, default=6, help="适配器层数")
    parser.add_argument("--adapter_len", type=int, default=5, help="适配器长度")
    parser.add_argument("--adapter_gate_scale", type=float, default=1.0, help="双适配器注意力门的放大系数；>1 可显式增强适配器通路")
    parser.add_argument("--adapter_dynamic_scale", type=float, default=0.0, help="按文本-时序分支分歧自适应放大双适配器门控的强度")
    parser.add_argument("--adapter_token_dynamic_scale", type=float, default=0.0, help="仅在局部高分歧patch/token上进一步放大双适配器门控，避免整段序列统一过注入")
    parser.add_argument("--adapter_layer_context_scale", type=float, default=0.0, help="让浅层适配器偏向近期局部上下文、深层适配器偏向全局上下文，匹配大模型分层语义")
    parser.add_argument("--patch_len", type=int, default=16, help="Patch长度")
    parser.add_argument("--stride", type=int, default=8, help="Patch步长")
    parser.add_argument("--p2t_topk_ratio", type=float, default=0.3, help="sanplm 中 P2T 锚点选择的 Top-K 比例")
    parser.add_argument("--p2t_low_rank", type=int, default=64, help="sanplm 中 P2T 低秩翻译秩")
    parser.add_argument("--p2t_translation_dropout", type=float, default=0.1, help="sanplm 中 P2T 翻译因子 dropout")
    parser.add_argument("--p2t_anchor_noise_scale", type=float, default=0.05, help="sanplm 中 P2T 翻译因子的温和高斯噪声强度")
    parser.add_argument("--soft_vocab_topk", type=int, default=8, help="sanplm 中伪文本候选词表 Top-K")
    parser.add_argument("--gumbel_tau", type=float, default=0.7, help="Gumbel-Softmax 初始温度")
    parser.add_argument("--gumbel_tau_end", type=float, default=0.1, help="Gumbel-Softmax 退火终止温度")
    parser.add_argument("--sanplm_pred_dropout", type=float, default=0.2, help="sanplm 预测头输入 dropout")
    parser.add_argument("--sanplm_adapter_coop_weight", type=float, default=0.05, help="sanplm 锚点协同余弦损失权重")
    parser.add_argument(
        "--sanplm_disable_ts_input_residual",
        action="store_true",
        default=False,
        help="禁用时序分支 patch 嵌入进入主干前的零初始化残差适配",
    )
    parser.add_argument("--sanplm_freeze_epochs", type=int, default=3, help="sanplm 前多少个 epoch 仅训练非 GPT 主干模块")
    parser.add_argument("--sanplm_unfreeze_last_n", type=int, default=2, help="sanplm 第二阶段解冻的 GPT block 数量")
    parser.add_argument(
        "--sanplm_unfreeze_pos_norm",
        action="store_true",
        default=False,
        help="sanplm 在 late_unfreeze 阶段额外解冻 GPT 位置嵌入和全部 LayerNorm（ln_1/ln_2/ln_f）",
    )
    parser.add_argument(
        "--sanplm_warmup_unfreeze_pos_norm",
        action="store_true",
        default=False,
        help="sanplm 在 adapter_warmup 阶段也额外解冻 GPT 位置嵌入和全部 LayerNorm（ln_1/ln_2/ln_f）",
    )
    parser.add_argument("--sanplm_syncbridge_layers", type=str, default="1,3,5", help="sanplm 前向共享状态桥插入的 GPT block 下标")
    parser.add_argument("--sanplm_syncbridge_kernel", type=int, default=3, help="sanplm 前向共享状态桥的因果平滑窗口")
    parser.add_argument("--sanplm_syncbridge_scale", type=float, default=0.2, help="sanplm 前向共享状态桥的全局混合强度")
    parser.add_argument("--sanplm_support_context_scale", type=float, default=0.65, help="sanplm 支撑证据桥中局部持续支持的占比")
    parser.add_argument("--sanplm_lagbridge_max_lag", type=int, default=2, help="sanplm lag bridge 可选的最大因果 patch lag")
    parser.add_argument("--sanplm_lagbridge_temperature", type=float, default=0.35, help="sanplm lag bridge 的软选择温度")
    parser.add_argument("--sanplm_bridge_decay_start", type=float, default=0.20, help="sanplm curriculum bridge 开始退火的训练进度")
    parser.add_argument("--sanplm_bridge_decay_floor", type=float, default=0.25, help="sanplm curriculum bridge 后期保留的最小桥强度比例")
    parser.add_argument("--sanplm_bridge_budget_floor", type=float, default=0.25, help="sanplm evidence-budget bridge 的最小共享预算")
    parser.add_argument("--sanplm_bridge_budget_init", type=float, default=0.55, help="sanplm evidence-budget bridge 的初始共享预算")
    parser.add_argument("--sanplm_bridge_budget_temp", type=float, default=0.70, help="sanplm evidence-budget bridge 的 patch 预算重分配温度")
    parser.add_argument("--sanplm_bridge_redistribute_mix", type=float, default=0.50, help="sanplm evidence-budget bridge 中原始门控与预算重分配门控的混合比例")
    parser.add_argument("--sanplm_bridge_transition_scale", type=float, default=0.35, help="sanplm evidence-budget bridge 对状态转移 patch 的额外预算偏置强度")
    parser.add_argument("--sanplm_anticollapse_agreement_center", type=float, default=0.72, help="sanplm anti-collapse bridge 最强共享的 agreement 中心")
    parser.add_argument("--sanplm_anticollapse_agreement_width", type=float, default=0.20, help="sanplm anti-collapse bridge 对 agreement 的容忍带宽")
    parser.add_argument("--sanplm_agreement_anchor_ema", type=float, default=0.98, help="sanplm agreement-memory bridge 中健康 agreement 锚点的 EMA 衰减")
    parser.add_argument("--sanplm_agreement_recovery_margin", type=float, default=0.04, help="sanplm agreement-memory bridge 中触发漂移恢复的最小偏移")
    parser.add_argument("--sanplm_agreement_recovery_scale", type=float, default=14.0, help="sanplm agreement-memory bridge 中漂移恢复的斜率")
    parser.add_argument("--sanplm_anchor_support_floor", type=float, default=0.35, help="sanplm agreement-memory bridge 中稳定 patch 的最小保桥比例")
    parser.add_argument("--sanplm_calibrate_progress", action="store_true", default=False, help="在验证集上为当前 checkpoint 选择最匹配的 bridge progress，再保存对应 progress")
    parser.add_argument("--sanplm_progress_candidates", type=str, default="0.0,0.1,0.2", help="sanplm progress 校准候选值，逗号分隔；会自动并入当前 progress")
    parser.add_argument("--sanplm_syncbridge_margin", type=float, default=0.12, help="sanplm 前向共享状态桥仅在稳定但有足够分歧时触发的余量阈值")
    parser.add_argument("--sanplm_syncbridge_topk", type=int, default=2, help="sanplm 前向共享状态桥每条 patch 序列保留的最高证据瓶颈数")
    parser.add_argument("--sanplm_context_anchor_kernel", type=int, default=5, help="sanplm 因果上下文锚点的 patch 级平滑窗口")
    parser.add_argument("--sanplm_context_anchor_scale", type=float, default=0.2, help="sanplm 因果上下文锚点注入强度")
    parser.add_argument("--sanplm_vocab_track_blend", type=float, default=0.35, help="sanplm 伪文本词义轨迹链中历史证据占比")
    parser.add_argument("--sanplm_vocab_track_bias", type=float, default=2.2, help="sanplm 伪文本词义轨迹链的稳定偏置")
    parser.add_argument("--sanplm_vocab_track_scale", type=float, default=6.0, help="sanplm 伪文本词义轨迹链对转移幅度的敏感度")
    parser.add_argument("--sanplm_semantic_conf_floor", type=float, default=0.20, help="sanplm 语义置信度的最小保留值")
    parser.add_argument("--sanplm_semantic_gap_scale", type=float, default=6.0, help="sanplm 候选词 top1-top2 间隔映射到语义置信度时的缩放")
    parser.add_argument("--sanplm_causal_rerank_scale", type=float, default=0.45, help="sanplm 用尾部因果上下文重排序词表候选的强度")
    parser.add_argument("--sanplm_causal_recent_window", type=int, default=3, help="sanplm 构造尾部因果上下文时使用的最近 patch 数")
    parser.add_argument("--sanplm_confidence_gate_floor", type=float, default=0.20, help="sanplm 低语义置信 patch 参与前向互感时保留的最小门控比例")
    parser.add_argument("--sanplm_p2t_cert_floor", type=float, default=0.25, help="sanplm 在 P2T 中保留低可靠 PTB 锚点的最小比例")
    parser.add_argument("--sanplm_p2t_support_scale", type=float, default=0.5, help="sanplm 在 P2T 证据认证中对因果支持度的使用强度")
    parser.add_argument("--sanplm_regime_span_mix", type=float, default=0.35, help="sanplm 在 P2T 中用稳定轨迹段权重替代固定半径扩散的混合比例")
    parser.add_argument("--sanplm_regime_span_scale", type=float, default=4.0, help="sanplm 在 P2T 中跨轨迹段断点的惩罚强度")
    parser.add_argument("--sanplm_anchor_stability_scale", type=float, default=0.35, help="sanplm 在 P2T 锚点打分中对稳定轨迹先验的使用强度")
    parser.add_argument("--sanplm_future_align_weight", type=float, default=0.02, help="sanplm 训练期让关键语义锚点对未来真实走势负责的辅助监督权重")
    parser.add_argument("--sanplm_transition_token_scale", type=float, default=0.45, help="sanplm 在 PTB 词表检索中用局部变化线索重排候选词的强度")
    parser.add_argument("--sanplm_ts_branch_depth", type=int, default=2, help="sanplm 中 TSB 停留的 GPT 浅层深度，用于显式做异构深度编码")
    parser.add_argument("--sanplm_branch_probe_weight", type=float, default=0.05, help="sanplm 训练时分支辅助预测损失权重，用于稳定双分支角色")
    parser.add_argument(
        "--sanplm_final_output_branch",
        type=str,
        default="dual",
        choices=["dual", "ts", "pt"],
        help="sanplm 最终预测头使用的输出分支：dual 为双分支拼接，ts/pt 为单分支消融",
    )
    parser.add_argument("--sanplm_focus_smooth_kernel", type=int, default=3, help="sanplm 共享关注锚点在 patch 维上的平滑窗口")
    parser.add_argument("--sanplm_focus_agreement_scale", type=float, default=0.6, help="sanplm 双分支共同关注锚点中的跨分支一致性权重")
    parser.add_argument("--sanplm_feedback_support_kernel", type=int, default=3, help="sanplm 锚点约束的 T2P 回写在 patch 维上的局部支持窗口")
    parser.add_argument("--sanplm_writeback_local_only", action="store_true", default=False, help="仅训练 writeback debias 新增的局部去偏支路")
    parser.add_argument(
        "--sanplm_preserve_user_hparams",
        action="store_true",
        help="对 sanplm.py 保留显式传入的 patch/lr/batch/patience/pred_dropout/coop_weight 超参，仅继续强制最小骨架相关配置",
    )
    parser.add_argument("--calf_text_tokens", type=int, default=4, help="每个样本的伪文本token数")
    parser.add_argument("--text_gate_min", type=float, default=0.1, help="文本门控最小保留比例")
    parser.add_argument("--t2t_patch_len", type=int, default=24, help="LLM-PS T2T patch 长度")
    parser.add_argument("--t2t_patch_stride", type=int, default=12, help="LLM-PS T2T patch 步长")
    parser.add_argument("--t2t_hidden_dim", type=int, default=96, help="LLM-PS T2T 隐层维度")
    parser.add_argument("--t2t_ff_dim", type=int, default=384, help="LLM-PS T2T 前馈层维度")
    parser.add_argument("--t2t_encoder_layers", type=int, default=4, help="LLM-PS T2T encoder 层数")
    parser.add_argument("--t2t_decoder_layers", type=int, default=1, help="LLM-PS T2T decoder 层数")
    parser.add_argument("--t2t_num_heads", type=int, default=4, help="LLM-PS T2T 注意力头数")
    parser.add_argument("--t2t_mask_ratio", type=float, default=0.75, help="LLM-PS T2T patch mask 比例")
    parser.add_argument("--t2t_loss_weight", type=float, default=0.05, help="LLM-PS T2T 辅助损失权重")
    parser.add_argument("--cib_weight", type=float, default=0.01, help="CIB总权重")
    parser.add_argument("--cib_pred_weight", type=float, default=0.1, help="CIB预测项权重")
    parser.add_argument("--cib_redundancy_weight", type=float, default=0.05, help="CIB冗余抑制项权重")
    parser.add_argument("--cib_kl_weight", type=float, default=0.001, help="CIB KL项权重")
    parser.add_argument("--cib_latent_dim", type=int, default=96, help="CIB瓶颈维度")
    parser.add_argument("--fusion_heads", type=int, default=12, help="CrossSpaceFusion多头数")
    parser.add_argument("--semantic_kernel_size", type=int, default=5, help="语义共振融合的因果卷积核大小")
    parser.add_argument("--text_fusion_max", type=float, default=0.35, help="文本语义最大注入比例")
    parser.add_argument("--semantic_residual_scale", type=float, default=0.35, help="语义残差预测支路输出到未来值的缩放比例")
    parser.add_argument("--history_skip_scale", type=float, default=0.5, help="历史直连线性外推支路输出到未来值的缩放比例")
    parser.add_argument("--fastpath_budget_scale", type=float, default=0.0, help="仅对 detail + semantic residual 快路径做预算收缩的最大强度")
    parser.add_argument("--fastpath_budget_hidden", type=int, default=32, help="快路径预算控制器隐层维度")
    parser.add_argument("--fastpath_min_scale", type=float, default=0.60, help="快路径预算收缩的最小保留比例")
    parser.add_argument("--residual_calibration_scale", type=float, default=0.0, help="仅对 semantic residual / history skip 做相对残差校准的最大缩放幅度")
    parser.add_argument("--residual_calibration_hidden", type=int, default=32, help="残差校准器隐层维度")
    parser.add_argument("--residual_calibration_start_ratio", type=float, default=0.25, help="残差校准器开始介入训练进度的比例，前期保持 scratch 基线轨迹")
    parser.add_argument("--text_smooth_weight", type=float, default=0.0, help="伪文本时间平滑正则权重")
    parser.add_argument("--detail_branch_layers", type=int, default=2, help="双分支模型中细节分支编码层数")
    parser.add_argument("--detail_kernel_size", type=int, default=5, help="双分支模型中细节分支局部卷积核大小")
    parser.add_argument("--detail_gate_min", type=float, default=0.05, help="细节分支门控最小值")
    parser.add_argument("--detail_gate_max", type=float, default=0.85, help="细节分支门控最大值")
    parser.add_argument("--band_smooth_weight", type=float, default=0.02, help="趋势分支平滑约束权重")
    parser.add_argument("--band_zero_weight", type=float, default=0.03, help="细节分支低频抑制约束权重")
    parser.add_argument("--band_orth_weight", type=float, default=0.01, help="双分支正交约束权重")
    parser.add_argument("--disable_band_constraints", action="store_true", default=False, help="禁用互补频带约束")
    parser.add_argument("--disable_detail_gate", action="store_true", default=False, help="禁用细节分支自适应门控")
    parser.add_argument("--fine_patch_len", type=int, default=8, help="双尺度细粒度分支patch长度")
    parser.add_argument("--fine_stride", type=int, default=4, help="双尺度细粒度分支patch步长")
    parser.add_argument("--coarse_patch_len", type=int, default=16, help="双尺度粗粒度分支patch长度")
    parser.add_argument("--coarse_stride", type=int, default=8, help="双尺度粗粒度分支patch步长")
    parser.add_argument("--branch_layers", type=int, default=3, help="双尺度分支编码层数")
    parser.add_argument("--branch_smooth_weight", type=float, default=0.02, help="粗分支平滑约束权重")
    parser.add_argument("--branch_lowfreq_weight", type=float, default=0.03, help="细分支低频抑制约束权重")
    parser.add_argument("--branch_consistency_weight", type=float, default=0.05, help="双尺度一致性约束权重")
    parser.add_argument("--branch_orth_weight", type=float, default=0.01, help="双尺度正交约束权重")
    parser.add_argument("--input_trend_scale", type=float, default=0.02, help="输入到语义检索与跨分支适配前的局部线性趋势先验注入比例")
    parser.add_argument("--input_detail_scale", type=float, default=0.0, help="输入到跨分支适配前的detrend细节先验注入比例")
    parser.add_argument("--branch_warmup_ratio", type=float, default=0.35, help="分支约束从弱到强的预热比例")
    parser.add_argument("--structure_align_weight", type=float, default=0.005, help="输入结构先验与双分支表征的一致性约束权重")
    parser.add_argument("--detail_fusion_min", type=float, default=1.0, help="细节预测输出融合的最小缩放比例")
    parser.add_argument("--detail_fusion_max", type=float, default=1.0, help="细节预测输出融合的最大缩放比例")
    parser.add_argument("--patch_trend_kernel", type=int, default=1, help="局部线性趋势先验在patch维上的平滑核大小")
    parser.add_argument("--history_trend_blend_max", type=float, default=0.0, help="输出端历史线性趋势校准的最大融合比例")
    parser.add_argument("--trend_anchor_scale", type=float, default=0.0, help="在高线性低细节patch上将趋势分支锚定回输入趋势先验的强度")
    parser.add_argument("--use_role_retrieval", action="store_true", default=False, help="启用基于趋势/转折/细节角色的伪文本检索")
    parser.add_argument("--use_role_adapter", action="store_true", default=False, help="启用基于角色摘要的双适配器条件token构造")
    parser.add_argument("--adapter_recent_bias", type=float, default=0.6, help="角色化适配器中对近期patch的强调强度")
    parser.add_argument("--adapter_transition_scale", type=float, default=0.75, help="角色化适配器中对转折patch的强调强度")
    parser.add_argument("--use_branch_route_input", action="store_true", default=False, help="启用patch级双分支角色路由，让文本分支与时序分支从输入主序列开始分工")
    parser.add_argument("--branch_route_scale", type=float, default=0.45, help="patch级双分支角色路由偏离原始patch嵌入的强度")
    parser.add_argument("--branch_route_adapter_scale", type=float, default=0.25, help="patch级双分支角色路由对跨分支adapter条件的额外注入强度")
    parser.add_argument("--branch_route_temperature", type=float, default=0.85, help="patch级双分支角色路由的softmax温度")
    parser.add_argument("--branch_route_start_ratio", type=float, default=0.0, help="patch级双分支角色路由开始介入训练进度的比例，前期可退化回 RoleBridge")
    parser.add_argument("--disable_branch_constraints", action="store_true", default=False, help="禁用双尺度分支约束")
    parser.add_argument("--disable_fusion_gate", action="store_true", default=False, help="禁用双尺度融合门控")
    parser.add_argument("--disable_context_align", action="store_true", default=False, help="禁用输入侧双尺度上下文对齐桥，用于严格消融")
    parser.add_argument("--state_prototypes", type=int, default=32, help="语义状态原型数量")
    parser.add_argument("--state_temperature", type=float, default=0.7, help="语义状态分配温度")
    parser.add_argument("--state_entropy_weight", type=float, default=0.001, help="语义状态低熵约束权重")
    parser.add_argument("--state_commitment_weight", type=float, default=0.01, help="语义状态原型贴合约束权重")
    parser.add_argument("--state_persistence_weight", type=float, default=0.01, help="语义状态时间持久性约束权重")
    parser.add_argument("--state_diversity_weight", type=float, default=0.01, help="语义状态原型多样性约束权重")
    
    # 对齐配置（主用跨模态对比学习 + Orth + CIB）
    parser.add_argument("--contrastive_weight", type=float, default=0.5, help="跨模态对比学习损失权重")
    parser.add_argument("--contrastive_temperature", type=float, default=0.07, help="跨模态对比学习温度")
    parser.add_argument("--alignment_weight", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--gw_weight", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--fgw_alpha", type=float, default=0.5, help=argparse.SUPPRESS)
    parser.add_argument("--gw_epsilon", type=float, default=0.08, help=argparse.SUPPRESS)
    parser.add_argument("--gw_iters", type=int, default=3, help=argparse.SUPPRESS)
    parser.add_argument("--sinkhorn_iters", type=int, default=15, help=argparse.SUPPRESS)
    parser.add_argument("--alignment_stage1_ratio", type=float, default=0.4, help="Stage1软对齐占总训练进度比例")
    parser.add_argument("--gw_ramp_ratio", type=float, default=0.2, help=argparse.SUPPRESS)
    parser.add_argument("--stage1_nce_weight", type=float, default=0.7, help=argparse.SUPPRESS)
    parser.add_argument("--stage1_cos_weight", type=float, default=0.3, help=argparse.SUPPRESS)
    parser.add_argument("--soft_align_temperature", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--hyper_weight", type=float, default=-1.0, help="Hypergraph Entanglement损失权重（<0时回退到gw_weight）")
    parser.add_argument("--hyper_temp", type=float, default=0.2, help="Hypergraph相似度温度")
    parser.add_argument("--sync_weight", type=float, default=0.3, help="Kuramoto同步损失权重")
    parser.add_argument("--kuramoto_k", type=float, default=0.5, help="Kuramoto耦合强度")
    parser.add_argument("--kuramoto_freq_std", type=float, default=0.1, help="Kuramoto自然频率标准差")
    parser.add_argument("--kuramoto_step", type=float, default=0.1, help="Kuramoto积分步长")
    parser.add_argument("--orth_weight", type=float, default=0.0, help="正交适配器正则权重（当前默认不启用）")
    parser.add_argument("--orth_rank", type=int, default=64, help="预训练注意力主子空间秩")
    parser.add_argument("--ts_mamba_layers", type=int, default=4, help="TS分支Mamba层数")
    parser.add_argument("--ts_mamba_expand", type=float, default=2.0, help="TS分支Mamba扩展倍率")
    parser.add_argument("--ts_mamba_conv_kernel", type=int, default=3, help="TS分支Mamba卷积核大小")
    parser.add_argument("--ts_encoder_layers", type=int, default=3, help="TS patch序列编码层数")
    parser.add_argument("--ts_encoder_dropout", type=float, default=0.1, help="TS patch序列编码dropout")
    parser.add_argument("--use_hyper_loss", action="store_true", default=False, help="启用Hypergraph辅助损失")
    parser.add_argument("--use_sync_loss", action="store_true", default=False, help="启用Kuramoto同步辅助损失")
    parser.add_argument("--use_orth_loss", action="store_true", default=False, help="启用正交辅助损失")
    parser.add_argument("--disable_cib_gate", action="store_true", default=False, help="禁用CIB文本门控")
    parser.add_argument("--enable_cib_gate", action="store_true", default=False, help="显式启用CIB文本门控")
    parser.add_argument("--disable_channel_mixer", action="store_true", default=False, help="禁用跨变量通道混合")
    parser.add_argument("--enable_channel_mixer", action="store_true", default=False, help="显式启用跨变量通道混合")
    parser.add_argument("--pretrained_ckpt", type=str, default="", help="可选：加载预训练checkpoint路径")
    parser.add_argument("--finetune_channel_mixer_only", action="store_true", default=False, help="仅微调通道混合参数")
    parser.add_argument("--finetune_adapter_only", action="store_true", default=False, help="仅微调双适配器查询与注意力门控参数")
    parser.add_argument("--finetune_readout_only", action="store_true", default=False, help="仅微调末端读出参数，适配新的输出语义")
    parser.add_argument("--finetune_bridge_only", action="store_true", default=False, help="仅微调时序-语义桥接相关参数，避免扰动预测头")
    parser.add_argument(
        "--bridge_include_pos_norm",
        action="store_true",
        default=False,
        help="在bridge_only基础上额外微调GPT位置嵌入和LayerNorm",
    )
    parser.add_argument(
        "--bridge_include_patch_head",
        action="store_true",
        default=False,
        help="在bridge_only基础上额外微调patch输入层、预测头和RevIN参数",
    )
    parser.add_argument("--eval_only", action="store_true", default=False, help="仅评估预训练checkpoint，不进行训练")
    parser.add_argument("--save_preds_npz", action="store_true", default=False, help="在评估或最终测试时额外保存 preds/trues，便于离线分析与集成")
    parser.add_argument("--eval_progress_override", type=float, default=None, help="评估时显式覆盖 checkpoint.progress；仅用于无训练的对照/导出")
    parser.add_argument("--analysis_export_dir", type=str, default="", help="可选：导出模型内部分析缓存；为空则不导出")
    parser.add_argument("--analysis_split", type=str, default="test", choices=["train", "val", "test"], help="导出分析缓存时使用的数据划分")
    parser.add_argument("--analysis_max_batches", type=int, default=1, help="导出分析缓存时最多保存的 batch 数")
    parser.add_argument("--analysis_filename", type=str, default="", help="导出分析缓存的文件名；为空则自动生成")
    parser.add_argument("--analysis_bridge_mode", type=str, default="selective", choices=["selective", "always_on"], help="导出分析缓存时 selective bridge 的分析模式；always_on 用于构造强融合对照")
    parser.add_argument("--ablate_evidence_chain", action="store_true", default=False, help="消融轨迹感知证据链，用即时 patch 查询替代")
    parser.add_argument("--ablate_selective_bridge", action="store_true", default=False, help="消融 selective bridge，仅保留双分支独立编码")
    parser.add_argument("--ablate_p2t", action="store_true", default=False, help="消融 P2T 适配器")

    # 优化配置
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载器工作进程数")
    parser.add_argument("--itr", type=int, default=1, help="实验次数")
    parser.add_argument("--train_epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="训练批大小")
    parser.add_argument("--patience", type=int, default=10, help="早停耐心")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="AdamW 权重衰减")
    parser.add_argument("--lradj", type=str, default="COS", help="学习率调整方式")
    parser.add_argument("--cos_tmax", type=int, default=0, help="Cosine T_max，<=0时使用train_epochs")
    parser.add_argument("--pct_start", type=float, default=0.2, help="OneCycleLR的pct_start")
    parser.add_argument("--ema_decay", type=float, default=0.0, help="训练期权重 EMA 衰减，0 表示关闭")
    parser.add_argument("--percent", type=int, default=100, help="训练数据百分比")
    parser.add_argument("--early_stop_metric", type=str, default="mse", choices=["mse", "mae"], help="早停监控指标")
    parser.add_argument("--checkpoint_metric", type=str, default="early_stop", choices=["early_stop", "vali_mse", "vali_mae", "test_mse", "test_mae"], help="checkpoint保存指标；early_stop表示沿用早停监控指标")
    parser.add_argument("--loss_switch_ratio", type=float, default=0.7, help="分阶段损失切换比例")
    parser.add_argument("--mse_loss_weight_early", type=float, default=0.3, help="前期MSE权重")
    parser.add_argument("--mse_loss_weight_late", type=float, default=0.7, help="后期MSE权重")
    parser.add_argument("--task_loss", type=str, default="mse", choices=["mse", "mix"], help="任务损失类型")
    parser.add_argument("--aux_warmup_epochs", type=int, default=3, help="辅助损失线性升权轮数")
    
    # 设备配置
    parser.add_argument("--use_gpu", type=bool, default=True, help="是否使用GPU")
    parser.add_argument("--gpu", type=int, default=0, help="GPU设备ID")
    parser.add_argument("--use_amp", action="store_true", default=False, help="是否使用混合精度训练")
    
    # 预生成数据配置
    parser.add_argument("--use_pregenerated", action="store_true", default=False, help="是否使用预生成的prompt和文本嵌入")
    parser.add_argument("--pregenerated_path", type=str, default="./checkpoints/pregenerated/", help="预生成数据的路径")

    args = parser.parse_args()
    return args


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_log_time_prefix():
    """返回日志创建时间前缀，精确到时分。"""
    return time.strftime("%H%M", time.localtime())


def normalize_model_name(args):
    """规范实验显示名称：优先使用 model_file 的文件名，不带 .py。"""
    model_file = str(getattr(args, "model_file", "") or "").strip()
    if model_file:
        args.model = os.path.splitext(os.path.basename(model_file))[0]
    return args


def load_model_class(model_file):
    """按需从模型文件动态加载 BALM_MedualTime 类。"""
    if not model_file:
        from BALM_MedualTime import BALM_MedualTime as DefaultBALM_MedualTime
        return DefaultBALM_MedualTime

    model_path = model_file if os.path.isabs(model_file) else os.path.abspath(model_file)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    module_name = f"codex_dynamic_model_{os.path.basename(model_path).replace('-', '_').replace('.', '_')}_{int(time.time() * 1000)}"
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load model spec from: {model_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model_class = getattr(module, "BALM_MedualTime", None)
    if model_class is None:
        raise AttributeError(f"BALM_MedualTime class not found in: {model_path}")
    return model_class


def model_forward(model, batch_x, mode, prompts=None, text_emb_init=None, alignment_progress=None, future_target=None):
    """兼容不同 BALM_MedualTime 版本的 forward 参数。"""
    if not hasattr(model, "_forward_param_names"):
        model._forward_param_names = set(inspect.signature(model.forward).parameters.keys())

    forward_kwargs = {"mode": mode}
    if "prompts" in model._forward_param_names:
        forward_kwargs["prompts"] = prompts
    if "text_emb_init" in model._forward_param_names:
        forward_kwargs["text_emb_init"] = text_emb_init
    if "alignment_progress" in model._forward_param_names:
        forward_kwargs["alignment_progress"] = alignment_progress
    if "future_target" in model._forward_param_names:
        forward_kwargs["future_target"] = future_target

    return model(batch_x, **forward_kwargs)


def is_sanplm_minimal_model(model_file):
    model_file_basename = os.path.basename(model_file) if model_file else ""
    return model_file_basename == "sanplm.py"


def is_sanplm_stage_managed_model(model_file):
    model_file_basename = os.path.basename(model_file) if model_file else ""
    return model_file_basename in (
        "sanplm.py",
        "model_dual.py",
        "sanplm_causalproto.py",
        "sanplm_causalvocab.py",
        "sanplm_causalvocab_syncbridge.py",
        "sanplm_causalvocab_commonstate.py",
        "sanplm_causalvocab_contextanchor.py",
        "sanplm_causalvocab_evidencechain.py",
        "sanplm_causalvocab_focusalign.py",
        "sanplm_causalvocab_feedbackclosure.py",
        "sanplm_causalvocab_selectivebridge.py",
        "sanplm_causalvocab_confidencebackoff.py",
        "sanplm_causalvocab_evidencererank.py",
        "sanplm_causalvocab_certifiedp2t.py",
        "sanplm_causalvocab_statechange.py",
        "sanplm_causalvocab_rolesplit.py",
        "sanplm_causalvocab_branchsupervision.py",
        "sanplm_causalvocab_horizonfusion.py",
        "sanplm_causalvocab_horizoncalibration.py",
        "sanplm_causalvocab_horizonprior.py",
        "sanplm_causalvocab_supportbridge.py",
        "sanplm_causalvocab_supportstatebridge.py",
        "sanplm_causalvocab_lagbridge.py",
        "sanplm_causalvocab_curriculumbridge.py",
        "sanplm_causalvocab_discrepancybridge.py",
        "sanplm_causalvocab_selectivebridge_semfreeze.py",
        "sanplm_causalvocab_selectivebridge_preddrop.py",
        "sanplm_causalvocab_persistentbridge.py",
        "sanplm_causalvocab_bottlenecktoken.py",
        "sanplm_causalvocab_persistentanchor.py",
        "sanplm_causalvocab_regimebridge.py",
        "sanplm_causalvocab_consensusbridge.py",
        "sanplm_causalvocab_anchorbridge.py",
        "sanplm_causalvocab_scaffoldbridge.py",
        "sanplm_causalvocab_horizonbridge.py",
        "sanplm_causalvocab_transitionbridge.py",
        "sanplm_causalvocab_tailbankbridge.py",
        "sanplm_causalvocab_echofreebridge.py",
        "sanplm_causalvocab_debiasedfeedback.py",
        "sanplm_causalvocab_debiasedrerank.py",
        "sanplm_causalvocab_trackspanbridge.py",
        "sanplm_causalvocab_futureguide.py",
        "sanplm_causalvocab_identitybridge.py",
        "sanplm_causalvocab_identitybridge_lite.py",
        "sanplm_causalvocab_continuationbridge.py",
        "sanplm_causalvocab_continuationbridge_soft.py",
        "sanplm_causalvocab_selectivebridge_anticollapse.py",
        "sanplm_causalvocab_selectivebridge_anticollapse_noblend.py",
        "sanplm_causalvocab_selectivebridge_anticollapse_nobridgeblend.py",
        "sanplm_causalvocab_selectivebridge_anticollapse_softt2p.py",
        "sanplm_causalvocab_selectivebridge_anticollapse_compactgate.py",
        "sanplm_causalvocab_selectivebridge_anticollapse_compacthparam.py",
        "sanplm_causalvocab_selectivebridge_anticollapse_noevidence.py",
        "sanplm_causalvocab_selectivebridge_tsonlyrefine.py",
        "sanplm_causalvocab_selectivebridge_agreementmemory.py",
        "sanplm_causalvocab_selectivebridge_privateguard.py",
        "sanplm_causalvocab_selectivebridge_privatehybrid.py",
        "sanplm_causalvocab_selectivebridge_closureaware.py",
        "sanplm_causalvocab_adapterrolefactor.py",
        "sanplm_causalvocab_semanticshield.py",
        "sanplm_causalvocab_adapterboundary.py",
        "sanplm_causalvocab_writebackdebias.py",
        "sanplm_causalvocab_selectivebridge_evidencebudget.py",
        "sanplm_causalvocab_selectivebridge_progresscalib.py",
    )


def apply_sanplm_minimal_recipe(args):
    if not is_sanplm_minimal_model(args.model_file):
        return False

    # 审稿人关注点：解决配置层冗余过多、创新归因不清晰的问题。
    args.adapter_layers = 4
    args.adapter_len = 8
    args.p2t_topk_ratio = 0.3
    args.p2t_low_rank = 64
    args.p2t_translation_dropout = 0.1
    args.p2t_anchor_noise_scale = 0.05
    args.soft_vocab_topk = 8
    args.task_loss = "mse"
    args.loss_switch_ratio = 0.0
    args.mse_loss_weight_early = 1.0
    args.mse_loss_weight_late = 1.0
    args.aux_warmup_epochs = 1

    if not getattr(args, "sanplm_preserve_user_hparams", False):
        args.patch_len = 24
        args.stride = 12
        args.sanplm_pred_dropout = 0.2
        args.sanplm_adapter_coop_weight = 0.05
        args.learning_rate = 0.0003
        args.batch_size = 32
        args.patience = 5

    float_zero_fields = (
        "cib_weight",
        "cib_pred_weight",
        "cib_redundancy_weight",
        "cib_kl_weight",
        "contrastive_weight",
        "state_entropy_weight",
        "state_commitment_weight",
        "state_persistence_weight",
        "state_diversity_weight",
        "branch_smooth_weight",
        "branch_lowfreq_weight",
        "branch_consistency_weight",
        "branch_orth_weight",
        "band_smooth_weight",
        "band_zero_weight",
        "band_orth_weight",
        "t2t_loss_weight",
        "text_smooth_weight",
        "gw_weight",
        "alignment_weight",
        "hyper_weight",
        "sync_weight",
        "orth_weight",
        "semantic_residual_scale",
        "history_skip_scale",
        "fastpath_budget_scale",
        "fastpath_min_scale",
        "residual_calibration_scale",
        "residual_calibration_start_ratio",
        "input_trend_scale",
        "input_detail_scale",
        "branch_warmup_ratio",
        "structure_align_weight",
        "history_trend_blend_max",
        "trend_anchor_scale",
        "branch_route_scale",
        "branch_route_adapter_scale",
        "branch_route_start_ratio",
        "text_fusion_max",
        "detail_gate_min",
        "detail_gate_max",
        "adapter_recent_bias",
        "adapter_transition_scale",
        "branch_route_temperature",
        "state_temperature",
        "contrastive_temperature",
        "fgw_alpha",
        "gw_epsilon",
        "alignment_stage1_ratio",
        "gw_ramp_ratio",
        "stage1_nce_weight",
        "stage1_cos_weight",
        "hyper_temp",
        "kuramoto_k",
        "kuramoto_freq_std",
        "kuramoto_step",
        "ts_mamba_expand",
        "ts_encoder_dropout",
        "text_gate_min",
        "t2t_mask_ratio",
    )
    for field in float_zero_fields:
        setattr(args, field, 0.0)

    int_zero_fields = (
        "detail_branch_layers",
        "branch_layers",
        "semantic_freq_topk",
        "semantic_hidden_dim",
        "decomp_kernel",
        "router_hidden_dim",
        "t2t_patch_len",
        "t2t_patch_stride",
        "t2t_hidden_dim",
        "t2t_ff_dim",
        "t2t_encoder_layers",
        "t2t_decoder_layers",
        "t2t_num_heads",
        "fusion_heads",
        "semantic_kernel_size",
        "fastpath_budget_hidden",
        "residual_calibration_hidden",
        "detail_kernel_size",
        "fine_patch_len",
        "fine_stride",
        "coarse_patch_len",
        "coarse_stride",
        "state_prototypes",
        "calf_text_tokens",
        "orth_rank",
        "ts_mamba_layers",
        "ts_encoder_layers",
        "ts_mamba_conv_kernel",
    )
    for field in int_zero_fields:
        setattr(args, field, 0)

    args.spectral_scales = "0.0"

    args.disable_band_constraints = True
    args.disable_detail_gate = True
    args.disable_branch_constraints = True
    args.disable_fusion_gate = True
    args.disable_context_align = True
    args.use_role_retrieval = False
    args.use_role_adapter = False
    args.use_branch_route_input = False
    args.enable_cib_gate = False
    args.disable_cib_gate = True
    args.enable_channel_mixer = False
    args.disable_channel_mixer = True
    args.use_hyper_loss = False
    args.use_sync_loss = False
    args.use_orth_loss = False

    return True


def maybe_apply_sanplm_train_stage(model, args, epoch, logger=None, force=False):
    if any(
        (
            getattr(args, "finetune_channel_mixer_only", False),
            getattr(args, "finetune_adapter_only", False),
            getattr(args, "finetune_readout_only", False),
            getattr(args, "finetune_bridge_only", False),
        )
    ):
        return None
    if not is_sanplm_stage_managed_model(args.model_file):
        return None
    if not hasattr(model, "set_train_stage"):
        return None

    stage_name = "adapter_warmup" if epoch < max(args.sanplm_freeze_epochs, 0) else "late_unfreeze"
    if not force and getattr(model, "_active_train_stage", None) == stage_name:
        return stage_name

    unfreeze_pos_norm = bool(getattr(args, "sanplm_unfreeze_pos_norm", False))
    if stage_name == "adapter_warmup":
        unfreeze_pos_norm = unfreeze_pos_norm and bool(
            getattr(args, "sanplm_warmup_unfreeze_pos_norm", False)
        )

    trainable_count = model.set_train_stage(
        stage_name=stage_name,
        unfreeze_last_n=max(0, int(args.sanplm_unfreeze_last_n)),
        unfreeze_pos_norm=unfreeze_pos_norm,
    )
    stage_message = (
        f"sanplm stage -> {stage_name} | "
        f"freeze_epochs={args.sanplm_freeze_epochs} | "
        f"unfreeze_last_n={args.sanplm_unfreeze_last_n} | "
        f"unfreeze_pos_norm={unfreeze_pos_norm} | "
        f"trainable_params={trainable_count}"
    )
    print(stage_message)
    if logger is not None:
        logger.info(stage_message)
    return stage_name


def get_task_loss_weights(epoch, train_epochs, switch_ratio, mse_weight_early, mse_weight_late):
    """按训练进度返回 L1/MSE 混合损失权重"""
    progress = float(epoch + 1) / float(max(train_epochs, 1))
    mse_weight = mse_weight_early if progress <= switch_ratio else mse_weight_late
    mse_weight = float(np.clip(mse_weight, 0.0, 1.0))
    l1_weight = 1.0 - mse_weight
    return l1_weight, mse_weight


def maybe_set_model_training_progress(model, progress):
    target_model = model.module if hasattr(model, "module") else model
    if hasattr(target_model, "set_training_progress"):
        target_model.set_training_progress(progress)


def build_progress_candidates(args, current_progress: float):
    candidates = [float(current_progress)]
    if not getattr(args, "sanplm_calibrate_progress", False):
        return candidates

    raw_candidates = str(getattr(args, "sanplm_progress_candidates", "")).split(",")
    for item in raw_candidates:
        item = item.strip()
        if not item:
            continue
        try:
            value = float(item)
        except ValueError:
            continue
        candidates.append(float(np.clip(value, 0.0, 1.0)))

    decay_start = getattr(args, "sanplm_bridge_decay_start", None)
    if decay_start is not None:
        candidates.append(float(np.clip(decay_start, 0.0, 1.0)))

    unique_candidates = []
    seen = set()
    for value in sorted(candidates):
        key = round(float(value), 6)
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(float(value))
    return unique_candidates


def select_progress_metrics(
    model,
    vali_loader,
    criterion,
    device,
    args,
    use_pregenerated,
    current_progress,
    monitor_key,
):
    best_bundle = None
    progress_candidates = build_progress_candidates(args, current_progress)

    for progress in progress_candidates:
        vali_loss, vali_preds, vali_trues = validate(
            model,
            vali_loader,
            criterion,
            device,
            args,
            use_pregenerated,
            alignment_progress=progress,
        )
        vali_mae, vali_mse, _, _, _ = metric(vali_preds, vali_trues)
        metric_pool = {
            "vali_mse": vali_mse,
            "vali_mae": vali_mae,
        }
        monitor_value = metric_pool[monitor_key]
        bundle = (
            float(progress),
            float(monitor_value),
            float(vali_loss),
            float(vali_mae),
            float(vali_mse),
        )
        if best_bundle is None or bundle[1] < best_bundle[1]:
            best_bundle = bundle

    return best_bundle, progress_candidates


def save_checkpoint_progress(checkpoint_path, progress: float):
    with open(f"{checkpoint_path}.progress", "w") as handle:
        handle.write(f"{float(progress):.8f}\n")


def load_checkpoint_progress(checkpoint_path, default: float = 1.0):
    progress_path = f"{checkpoint_path}.progress"
    if not os.path.exists(progress_path):
        return float(default)
    try:
        with open(progress_path, "r") as handle:
            return float(handle.read().strip())
    except (OSError, ValueError):
        return float(default)


class EMAHelper:
    """训练期参数平滑：用于保留早期较优的稳定权重轨迹。"""

    def __init__(self, model, decay: float):
        self.decay = float(decay)
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.backup = {}

    def update(self, model):
        one_minus_decay = 1.0 - self.decay
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=one_minus_decay)

    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


def train_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    criterion_l1,
    criterion_mse,
    device,
    args,
    epoch,
    ema_helper=None,
    use_pregenerated=False,
):
    """训练一个epoch"""
    model.train()
    train_loss = []
    train_l1_losses = []
    train_mse_losses = []
    alignment_losses = []
    probe_losses = []
    if args.task_loss == "mix":
        l1_weight, mse_weight = get_task_loss_weights(
            epoch,
            args.train_epochs,
            args.loss_switch_ratio,
            args.mse_loss_weight_early,
            args.mse_loss_weight_late,
        )
    else:
        l1_weight, mse_weight = 0.0, 1.0
    aux_scale = min(1.0, float(epoch + 1) / float(max(args.aux_warmup_epochs, 1)))
    alignment_progress = float(epoch + 1) / float(max(args.train_epochs, 1))
    maybe_set_model_training_progress(model, alignment_progress)
    
    for i, batch_data in enumerate(train_loader):
        optimizer.zero_grad()
        
        # 解包数据
        if use_pregenerated:
            batch_x, batch_y, batch_x_mark, batch_y_mark, prompts, text_embeddings = batch_data
            # prompts 是 List[List[str]], text_embeddings 是 List[np.ndarray]
            # 需要将它们展平成 B*N 的形式
            B = batch_x.shape[0]
            N = batch_x.shape[2]
            
            # 展平 prompts: List[List[str]] -> List[str]
            prompts_flat = []
            for sample_prompts in prompts:
                prompts_flat.extend(sample_prompts)
            
            # 展平 text_embeddings: List[np.ndarray] -> np.ndarray [B*N, 768]
            text_embeddings_flat = np.concatenate(text_embeddings, axis=0)
        else:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data
            prompts_flat = None
            text_embeddings_flat = None
        
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        
        # 前向传播 - 模型输出 [batch, pred_len, n_vars]
        if use_pregenerated:
            model_outputs = model_forward(
                model,
                batch_x,
                mode='train',
                prompts=prompts_flat,
                text_emb_init=text_embeddings_flat,
                alignment_progress=alignment_progress,
                future_target=batch_y[:, -args.pred_len:, :],
            )
        else:
            model_outputs = model_forward(
                model,
                batch_x,
                mode='train',
                alignment_progress=alignment_progress,
                future_target=batch_y[:, -args.pred_len:, :],
            )

        extra_outputs = None
        if isinstance(model_outputs, tuple):
            if len(model_outputs) >= 3:
                outputs, alignment_loss, extra_outputs = model_outputs[:3]
            else:
                outputs, alignment_loss = model_outputs
        else:
            outputs = model_outputs
            alignment_loss = outputs.new_tensor(0.0)
        
        # 计算任务损失
        f_dim = -1 if args.features == "MS" else 0
        # 获取目标: batch_y 的最后 pred_len 个时间步
        target = batch_y[:, -args.pred_len:, f_dim:]
        
        # 如果是MS任务，只预测最后一个变量
        if args.features == "MS":
            outputs = outputs[:, :, f_dim:]
        
        task_l1 = criterion_l1(outputs, target)
        task_mse = criterion_mse(outputs, target)
        if args.task_loss == "mix":
            task_loss = l1_weight * task_l1 + mse_weight * task_mse
        else:
            task_loss = task_mse

        probe_loss = outputs.new_tensor(0.0)
        if isinstance(extra_outputs, dict):
            ts_probe = extra_outputs.get("ts_probe")
            pt_probe = extra_outputs.get("pt_probe")
            if ts_probe is not None:
                if args.features == "MS":
                    ts_probe = ts_probe[:, :, f_dim:]
                probe_loss = probe_loss + criterion_mse(ts_probe, target)
            if pt_probe is not None:
                if args.features == "MS":
                    pt_probe = pt_probe[:, :, f_dim:]
                probe_loss = probe_loss + criterion_mse(pt_probe, target)
            probe_loss = args.sanplm_branch_probe_weight * probe_loss
        
        # 总损失 = 任务损失 + 对齐损失 + 分支辅助预测损失
        loss = task_loss + aux_scale * alignment_loss + probe_loss
        
        loss.backward()
        optimizer.step()
        if ema_helper is not None:
            ema_helper.update(model)
        if args.lradj != "COS":
            scheduler.step()
        
        train_loss.append(task_loss.item())
        train_l1_losses.append(task_l1.item())
        train_mse_losses.append(task_mse.item())
        alignment_losses.append((aux_scale * alignment_loss).item())
        probe_losses.append(probe_loss.item())
        
    return (
        np.average(train_loss),
        np.average(alignment_losses),
        np.average(probe_losses),
        np.average(train_l1_losses),
        np.average(train_mse_losses),
        l1_weight,
        mse_weight,
    )


def validate(model, vali_loader, criterion, device, args, use_pregenerated=False, alignment_progress=None):
    """验证函数"""
    model.eval()
    if alignment_progress is not None:
        maybe_set_model_training_progress(model, alignment_progress)
    total_loss = []
    preds = []
    trues = []
    
    with torch.no_grad():
        for i, batch_data in enumerate(vali_loader):
            # 解包数据
            if use_pregenerated:
                batch_x, batch_y, batch_x_mark, batch_y_mark, prompts, text_embeddings = batch_data
                # 展平 prompts 和 text_embeddings
                B = batch_x.shape[0]
                N = batch_x.shape[2]
                
                prompts_flat = []
                for sample_prompts in prompts:
                    prompts_flat.extend(sample_prompts)
                
                text_embeddings_flat = np.concatenate(text_embeddings, axis=0)
            else:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data
                prompts_flat = None
                text_embeddings_flat = None
            
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            
            # 前向传播 - 模型输出 [batch, pred_len, n_vars]
            if use_pregenerated:
                outputs = model_forward(
                    model,
                    batch_x,
                    mode='eval',
                    prompts=prompts_flat,
                    text_emb_init=text_embeddings_flat,
                    alignment_progress=alignment_progress,
                )
            else:
                outputs = model_forward(model, batch_x, mode='eval', alignment_progress=alignment_progress)
            
            # 计算损失
            f_dim = -1 if args.features == "MS" else 0
            target = batch_y[:, -args.pred_len:, f_dim:]
            
            if args.features == "MS":
                outputs = outputs[:, :, f_dim:]
            
            loss = criterion(outputs, target)
            
            preds.append(outputs.cpu().numpy())
            trues.append(target.cpu().numpy())
            total_loss.append(loss.item())
    
    total_loss = np.average(total_loss)
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    
    model.train()
    return total_loss, preds, trues


def resolve_analysis_export_path(path, setting, args, split_name):
    export_root = str(getattr(args, "analysis_export_dir", "") or "").strip()
    if export_root:
        export_root = export_root if os.path.isabs(export_root) else os.path.abspath(export_root)
    else:
        export_root = path
    os.makedirs(export_root, exist_ok=True)

    filename = str(getattr(args, "analysis_filename", "") or "").strip()
    if not filename:
        filename = f"{setting}_analysis_{split_name}_s{args.seed}.pt"
    return os.path.join(export_root, filename)


def maybe_export_analysis(
    model,
    train_loader,
    vali_loader,
    test_loader,
    device,
    args,
    use_pregenerated,
    alignment_progress,
    path,
    setting,
):
    export_flag = str(getattr(args, "analysis_export_dir", "") or "").strip()
    filename_flag = str(getattr(args, "analysis_filename", "") or "").strip()
    if not export_flag and not filename_flag:
        return None

    target_model = model.module if hasattr(model, "module") else model
    if not hasattr(target_model, "enable_analysis_cache"):
        print("Warning: current model does not support analysis export; skipping.")
        return None
    if hasattr(target_model, "set_analysis_bridge_mode"):
        target_model.set_analysis_bridge_mode(getattr(args, "analysis_bridge_mode", "selective"))

    split_name = str(getattr(args, "analysis_split", "test"))
    loader_map = {
        "train": train_loader,
        "val": vali_loader,
        "test": test_loader,
    }
    data_loader = loader_map[split_name]
    export_path = resolve_analysis_export_path(path, setting, args, split_name)
    max_batches = max(1, int(getattr(args, "analysis_max_batches", 1)))

    if alignment_progress is not None:
        maybe_set_model_training_progress(model, alignment_progress)

    was_training = model.training
    model.eval()
    target_model.enable_analysis_cache(True, move_to_cpu=True)
    records = []

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            if batch_idx >= max_batches:
                break

            if use_pregenerated:
                batch_x, batch_y, batch_x_mark, batch_y_mark, prompts, text_embeddings = batch_data
                prompts_flat = []
                for sample_prompts in prompts:
                    prompts_flat.extend(sample_prompts)
                text_embeddings_flat = np.concatenate(text_embeddings, axis=0)
            else:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data
                prompts_flat = None
                text_embeddings_flat = None

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            if use_pregenerated:
                outputs = model_forward(
                    model,
                    batch_x,
                    mode="eval",
                    prompts=prompts_flat,
                    text_emb_init=text_embeddings_flat,
                    alignment_progress=alignment_progress,
                )
            else:
                outputs = model_forward(
                    model,
                    batch_x,
                    mode="eval",
                    alignment_progress=alignment_progress,
                )

            f_dim = -1 if args.features == "MS" else 0
            target = batch_y[:, -args.pred_len:, f_dim:]
            if args.features == "MS":
                outputs = outputs[:, :, f_dim:]

            records.append(
                {
                    "batch_index": batch_idx,
                    "input": batch_x.detach().cpu(),
                    "target": target.detach().cpu(),
                    "prediction": outputs.detach().cpu(),
                    "analysis": getattr(target_model, "_last_analysis", {}),
                }
            )

    target_model.enable_analysis_cache(False)
    model.train(was_training)
    payload = {
        "setting": setting,
        "split": split_name,
        "seed": int(args.seed),
        "alignment_progress": None if alignment_progress is None else float(alignment_progress),
        "analysis_bridge_mode": str(getattr(args, "analysis_bridge_mode", "selective")),
        "args": dict(vars(args)),
        "records": records,
    }
    torch.save(payload, export_path)
    print(f"Analysis export saved to {export_path}")
    return export_path


def main():
    """主函数"""
    args = get_args()
    args = normalize_model_name(args)
    sanplm_recipe_applied = apply_sanplm_minimal_recipe(args)
    set_seed(args.seed)
    
    # 设备配置：直接尝试 CUDA 分配，避免 device_count 触发的环境兼容问题
    if args.use_gpu:
        try:
            device = torch.device(f"cuda:{args.gpu}")
            _ = torch.zeros(1, device=device)
            try:
                torch.cuda.manual_seed(args.seed)
                torch.cuda.manual_seed_all(args.seed)
            except Exception:
                pass
            print(f"Using GPU: {args.gpu}")
        except Exception as exc:
            device = torch.device("cpu")
            print(f"Using CPU (CUDA init failed: {exc})")
    else:
        device = torch.device("cpu")
        print("Using CPU (--use_gpu is False)")
    
    for ii in range(args.itr):
        # 设置实验记录
        log_time_prefix = get_log_time_prefix()
        setting = (
            f"{log_time_prefix}_{args.model}_{args.data}_"
            f"ft{args.features}_sl{args.seq_len}_pl{args.pred_len}_{ii}"
        )
        
        # 创建检查点目录
        path = os.path.join(args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # 设置日志
        log_filename = f"{log_time_prefix}_record_s{args.seed}.log"
        logger = get_logger(path, f"{__name__}.{setting}", log_filename)
        logger.info(args)
        if sanplm_recipe_applied:
            recipe_prefix = "Applied sanplm minimal recipe"
            if args.sanplm_preserve_user_hparams:
                recipe_prefix += " with user hparam overrides"
            logger.info(
                f"{recipe_prefix}: disable legacy losses/constraints, "
                f"patch_len={args.patch_len}, stride={args.stride}, lr={args.learning_rate}, "
                f"batch_size={args.batch_size}, patience={args.patience}, "
                f"freeze_epochs={args.sanplm_freeze_epochs}, unfreeze_last_n={args.sanplm_unfreeze_last_n}, "
                f"pred_dropout={args.sanplm_pred_dropout}, coop_weight={args.sanplm_adapter_coop_weight}."
            )
        
        # 加载数据
        print("Loading data...")
        train_data, train_loader = data_provider(args, "train", multivariate=True)
        vali_data, vali_loader = data_provider(args, "val", multivariate=True)
        test_data, test_loader = data_provider(args, "test", multivariate=True)
        print(f"Train samples: {len(train_data)}, Val samples: {len(vali_data)}, Test samples: {len(test_data)}")
        
        # 创建模型配置
        from types import SimpleNamespace
        ts_config = SimpleNamespace(
            context_points=args.seq_len,
            patch_len=args.patch_len,
            stride=args.stride,
            vars=args.enc_in,
            revin=False
        )
        
        from transformers import GPT2Config
        contrastive_weight = args.contrastive_weight
        if args.gw_weight is not None:
            contrastive_weight = args.gw_weight
        elif args.alignment_weight is not None:
            contrastive_weight = args.alignment_weight

        contrastive_temperature = args.contrastive_temperature
        if args.soft_align_temperature is not None:
            contrastive_temperature = args.soft_align_temperature

        hyper_weight = contrastive_weight if args.hyper_weight < 0 else args.hyper_weight
        try:
            spectral_scale_factors = tuple(
                float(v.strip()) for v in args.spectral_scales.split(",") if v.strip()
            )
            if not spectral_scale_factors:
                spectral_scale_factors = (0.5, 1.0, 2.0)
        except ValueError:
            print(f"Invalid --spectral_scales={args.spectral_scales}, fallback to default 0.5,1.0,2.0")
            spectral_scale_factors = (0.5, 1.0, 2.0)
        model_file_basename = os.path.basename(args.model_file) if args.model_file else ""
        is_sanplm_minimal = model_file_basename == "sanplm.py"
        is_sanplm_causalproto = model_file_basename in (
            "model_dual.py",
            "sanplm_causalproto.py",
            "sanplm_causalvocab.py",
            "sanplm_causalvocab_syncbridge.py",
            "sanplm_causalvocab_commonstate.py",
            "sanplm_causalvocab_contextanchor.py",
            "sanplm_causalvocab_evidencechain.py",
            "sanplm_causalvocab_focusalign.py",
            "sanplm_causalvocab_feedbackclosure.py",
            "sanplm_causalvocab_selectivebridge.py",
            "sanplm_causalvocab_confidencebackoff.py",
            "sanplm_causalvocab_evidencererank.py",
            "sanplm_causalvocab_certifiedp2t.py",
            "sanplm_causalvocab_statechange.py",
            "sanplm_causalvocab_rolesplit.py",
            "sanplm_causalvocab_branchsupervision.py",
            "sanplm_causalvocab_horizonfusion.py",
            "sanplm_causalvocab_horizoncalibration.py",
            "sanplm_causalvocab_horizonprior.py",
            "sanplm_causalvocab_supportbridge.py",
            "sanplm_causalvocab_supportstatebridge.py",
            "sanplm_causalvocab_lagbridge.py",
            "sanplm_causalvocab_curriculumbridge.py",
            "sanplm_causalvocab_discrepancybridge.py",
            "sanplm_causalvocab_selectivebridge_semfreeze.py",
            "sanplm_causalvocab_selectivebridge_preddrop.py",
            "sanplm_causalvocab_persistentbridge.py",
            "sanplm_causalvocab_bottlenecktoken.py",
            "sanplm_causalvocab_persistentanchor.py",
            "sanplm_causalvocab_regimebridge.py",
            "sanplm_causalvocab_consensusbridge.py",
            "sanplm_causalvocab_anchorbridge.py",
            "sanplm_causalvocab_scaffoldbridge.py",
            "sanplm_causalvocab_horizonbridge.py",
            "sanplm_causalvocab_transitionbridge.py",
            "sanplm_causalvocab_tailbankbridge.py",
            "sanplm_causalvocab_echofreebridge.py",
            "sanplm_causalvocab_debiasedfeedback.py",
            "sanplm_causalvocab_debiasedrerank.py",
            "sanplm_causalvocab_trackspanbridge.py",
            "sanplm_causalvocab_futureguide.py",
            "sanplm_causalvocab_identitybridge.py",
            "sanplm_causalvocab_identitybridge_lite.py",
            "sanplm_causalvocab_continuationbridge.py",
            "sanplm_causalvocab_continuationbridge_soft.py",
            "sanplm_causalvocab_selectivebridge_anticollapse.py",
            "sanplm_causalvocab_selectivebridge_anticollapse_noblend.py",
            "sanplm_causalvocab_selectivebridge_anticollapse_nobridgeblend.py",
            "sanplm_causalvocab_selectivebridge_anticollapse_softt2p.py",
            "sanplm_causalvocab_selectivebridge_anticollapse_compactgate.py",
            "sanplm_causalvocab_selectivebridge_anticollapse_compacthparam.py",
            "sanplm_causalvocab_selectivebridge_tsonlyrefine.py",
            "sanplm_causalvocab_selectivebridge_agreementmemory.py",
            "sanplm_causalvocab_selectivebridge_privateguard.py",
            "sanplm_causalvocab_selectivebridge_privatehybrid.py",
            "sanplm_causalvocab_selectivebridge_closureaware.py",
            "sanplm_causalvocab_adapterrolefactor.py",
            "sanplm_causalvocab_semanticshield.py",
            "sanplm_causalvocab_adapterboundary.py",
            "sanplm_causalvocab_writebackdebias.py",
            "sanplm_causalvocab_selectivebridge_evidencebudget.py",
            "sanplm_causalvocab_selectivebridge_progresscalib.py",
        )
        config = GPT2Config(
            use_cache=False,
            n_layer=6,  # GPT2 总层数设置为 6
            num_hidden_layers=6,  # 显式同步，防止 BALM_MedualTime.py 中使用此参数时出错
            d_model=args.d_model,
            adapter_layers=args.adapter_layers,
            adapter_len=args.adapter_len,
            adapter_gate_scale=args.adapter_gate_scale,
            adapter_dynamic_scale=args.adapter_dynamic_scale,
            adapter_token_dynamic_scale=args.adapter_token_dynamic_scale,
            adapter_layer_context_scale=args.adapter_layer_context_scale,
            adapter_head=12,
            attn_pdrop=args.dropout,
            embd_pdrop=args.dropout,
            resid_pdrop=args.dropout,
            ts_config=ts_config,
            pred_len=args.pred_len,
            alignment_weight=contrastive_weight,
            contrastive_weight=contrastive_weight,
            fgw_alpha=args.fgw_alpha,
            gw_epsilon=args.gw_epsilon,
            gw_iters=args.gw_iters,
            sinkhorn_iters=args.sinkhorn_iters,
            alignment_stage1_ratio=args.alignment_stage1_ratio,
            gw_ramp_ratio=args.gw_ramp_ratio,
            stage1_nce_weight=args.stage1_nce_weight,
            stage1_cos_weight=args.stage1_cos_weight,
            soft_align_temperature=contrastive_temperature,
            contrastive_temperature=contrastive_temperature,
            hyper_weight=hyper_weight,
            hyper_orders=(3, 4, 5),
            hyper_temp=args.hyper_temp,
            kuramoto_k=args.kuramoto_k,
            kuramoto_freq_std=args.kuramoto_freq_std,
            kuramoto_step=args.kuramoto_step,
            sync_weight=args.sync_weight,
            orth_weight=args.orth_weight,
            orth_rank=args.orth_rank,
            calf_text_tokens=args.calf_text_tokens,
            text_gate_min=args.text_gate_min,
            t2t_patch_len=args.t2t_patch_len,
            t2t_patch_stride=args.t2t_patch_stride,
            t2t_hidden_dim=args.t2t_hidden_dim,
            t2t_ff_dim=args.t2t_ff_dim,
            t2t_encoder_layers=args.t2t_encoder_layers,
            t2t_decoder_layers=args.t2t_decoder_layers,
            t2t_num_heads=args.t2t_num_heads,
            t2t_mask_ratio=args.t2t_mask_ratio,
            t2t_loss_weight=args.t2t_loss_weight,
            cib_weight=args.cib_weight,
            cib_pred_weight=args.cib_pred_weight,
            cib_redundancy_weight=args.cib_redundancy_weight,
            cib_kl_weight=args.cib_kl_weight,
            cib_latent_dim=args.cib_latent_dim,
            fusion_heads=args.fusion_heads,
            semantic_kernel_size=args.semantic_kernel_size,
            text_fusion_max=args.text_fusion_max,
            semantic_residual_scale=args.semantic_residual_scale,
            history_skip_scale=args.history_skip_scale,
            fastpath_budget_scale=args.fastpath_budget_scale,
            fastpath_budget_hidden=args.fastpath_budget_hidden,
            fastpath_min_scale=args.fastpath_min_scale,
            residual_calibration_scale=args.residual_calibration_scale,
            residual_calibration_hidden=args.residual_calibration_hidden,
            residual_calibration_start_ratio=args.residual_calibration_start_ratio,
            text_smooth_weight=args.text_smooth_weight,
            decomp_kernel=args.decomp_kernel,
            router_hidden_dim=args.router_hidden_dim,
            ts_mamba_layers=args.ts_mamba_layers,
            ts_mamba_expand=args.ts_mamba_expand,
            ts_mamba_conv_kernel=args.ts_mamba_conv_kernel,
            ts_encoder_layers=args.ts_encoder_layers,
            ts_encoder_dropout=args.ts_encoder_dropout,
            semantic_freq_topk=args.semantic_freq_topk,
            semantic_hidden_dim=args.semantic_hidden_dim,
            spectral_scale_factors=spectral_scale_factors,
            use_hyper_loss=args.use_hyper_loss,
            use_sync_loss=args.use_sync_loss,
            use_orth_loss=args.use_orth_loss,
            use_cib_gate=args.enable_cib_gate or not args.disable_cib_gate,
            use_channel_mixer=args.enable_channel_mixer and not args.disable_channel_mixer,
            detail_branch_layers=args.detail_branch_layers,
            detail_kernel_size=args.detail_kernel_size,
            detail_gate_min=args.detail_gate_min,
            detail_gate_max=args.detail_gate_max,
            band_smooth_weight=args.band_smooth_weight,
            band_zero_weight=args.band_zero_weight,
            band_orth_weight=args.band_orth_weight,
            use_band_constraints=not args.disable_band_constraints,
            use_detail_gate=not args.disable_detail_gate,
            fine_patch_len=args.fine_patch_len,
            fine_stride=args.fine_stride,
            coarse_patch_len=args.coarse_patch_len,
            coarse_stride=args.coarse_stride,
            branch_layers=args.branch_layers,
            branch_smooth_weight=args.branch_smooth_weight,
            branch_lowfreq_weight=args.branch_lowfreq_weight,
            branch_consistency_weight=args.branch_consistency_weight,
            branch_orth_weight=args.branch_orth_weight,
            input_trend_scale=args.input_trend_scale,
            input_detail_scale=args.input_detail_scale,
            branch_warmup_ratio=args.branch_warmup_ratio,
            structure_align_weight=args.structure_align_weight,
            detail_fusion_min=args.detail_fusion_min,
            detail_fusion_max=args.detail_fusion_max,
            patch_trend_kernel=args.patch_trend_kernel,
            history_trend_blend_max=args.history_trend_blend_max,
            trend_anchor_scale=args.trend_anchor_scale,
            use_role_retrieval=args.use_role_retrieval,
            use_role_adapter=args.use_role_adapter,
            adapter_recent_bias=args.adapter_recent_bias,
            adapter_transition_scale=args.adapter_transition_scale,
            use_branch_route_input=args.use_branch_route_input,
            branch_route_scale=args.branch_route_scale,
            branch_route_adapter_scale=args.branch_route_adapter_scale,
            branch_route_temperature=args.branch_route_temperature,
            branch_route_start_ratio=args.branch_route_start_ratio,
            use_branch_constraints=not args.disable_branch_constraints,
            use_fusion_gate=not args.disable_fusion_gate,
            use_context_align=not args.disable_context_align,
            state_prototypes=args.state_prototypes,
            state_temperature=args.state_temperature,
            state_entropy_weight=args.state_entropy_weight,
            state_commitment_weight=args.state_commitment_weight,
            state_persistence_weight=args.state_persistence_weight,
            state_diversity_weight=args.state_diversity_weight,
            final_output_branch=args.sanplm_final_output_branch,
            data_name=args.data,
            root_path=args.root_path,
            data_path=args.data_path,
            features=args.features,
            target=args.target,
            percent=args.percent,
            cache_batch_size=args.batch_size,
            experiment_seed=args.seed,
            centroid_cache_dir=os.path.join(args.checkpoints, "centroids"),
            mode_clusters=32,
            p2t_topk=5 if is_sanplm_causalproto else 3,
            p2t_topk_ratio=args.p2t_topk_ratio if is_sanplm_minimal else 0.0,
            p2t_low_rank=args.p2t_low_rank,
            p2t_anchor_radius=args.stride * 2 if is_sanplm_causalproto else args.stride,
            p2t_translation_dropout=args.p2t_translation_dropout,
            p2t_anchor_noise_scale=args.p2t_anchor_noise_scale,
            soft_vocab_topk=args.soft_vocab_topk if (is_sanplm_minimal or is_sanplm_causalproto) else 16,
            gumbel_tau=args.gumbel_tau if not is_sanplm_causalproto else 0.5,
            gumbel_tau_end=args.gumbel_tau_end,
            use_straight_through_tokens=True,
            max_centroid_patches=200000,
            sanplm_pred_dropout=args.sanplm_pred_dropout,
            adapter_coop_weight=args.sanplm_adapter_coop_weight,
            disable_ts_input_residual=args.sanplm_disable_ts_input_residual,
            disable_evidence_chain=args.ablate_evidence_chain,
            disable_selective_bridge=args.ablate_selective_bridge,
            disable_p2t=args.ablate_p2t,
            syncbridge_layers=args.sanplm_syncbridge_layers,
            syncbridge_kernel=args.sanplm_syncbridge_kernel,
            syncbridge_scale=args.sanplm_syncbridge_scale,
            support_context_scale=args.sanplm_support_context_scale,
            lagbridge_max_lag=args.sanplm_lagbridge_max_lag,
            lagbridge_temperature=args.sanplm_lagbridge_temperature,
            bridge_decay_start=args.sanplm_bridge_decay_start,
            bridge_decay_floor=args.sanplm_bridge_decay_floor,
            bridge_budget_floor=args.sanplm_bridge_budget_floor,
            bridge_budget_init=args.sanplm_bridge_budget_init,
            bridge_budget_temp=args.sanplm_bridge_budget_temp,
            bridge_redistribute_mix=args.sanplm_bridge_redistribute_mix,
            bridge_transition_scale=args.sanplm_bridge_transition_scale,
            anticollapse_agreement_center=args.sanplm_anticollapse_agreement_center,
            anticollapse_agreement_width=args.sanplm_anticollapse_agreement_width,
            agreement_anchor_ema=args.sanplm_agreement_anchor_ema,
            agreement_recovery_margin=args.sanplm_agreement_recovery_margin,
            agreement_recovery_scale=args.sanplm_agreement_recovery_scale,
            anchor_support_floor=args.sanplm_anchor_support_floor,
            syncbridge_margin=args.sanplm_syncbridge_margin,
            syncbridge_topk=args.sanplm_syncbridge_topk,
            context_anchor_kernel=args.sanplm_context_anchor_kernel,
            context_anchor_scale=args.sanplm_context_anchor_scale,
            vocab_track_blend=args.sanplm_vocab_track_blend,
            vocab_track_bias=args.sanplm_vocab_track_bias,
            vocab_track_scale=args.sanplm_vocab_track_scale,
            semantic_conf_floor=args.sanplm_semantic_conf_floor,
            semantic_gap_scale=args.sanplm_semantic_gap_scale,
            causal_rerank_scale=args.sanplm_causal_rerank_scale,
            causal_recent_window=args.sanplm_causal_recent_window,
            confidence_gate_floor=args.sanplm_confidence_gate_floor,
            p2t_cert_floor=args.sanplm_p2t_cert_floor,
            p2t_support_scale=args.sanplm_p2t_support_scale,
            regime_span_mix=args.sanplm_regime_span_mix,
            regime_span_scale=args.sanplm_regime_span_scale,
            anchor_stability_scale=args.sanplm_anchor_stability_scale,
            future_align_weight=args.sanplm_future_align_weight,
            transition_token_scale=args.sanplm_transition_token_scale,
            ts_branch_depth=args.sanplm_ts_branch_depth,
            branch_probe_weight=args.sanplm_branch_probe_weight,
            focus_smooth_kernel=args.sanplm_focus_smooth_kernel,
            focus_agreement_scale=args.sanplm_focus_agreement_scale,
            feedback_support_kernel=args.sanplm_feedback_support_kernel,
            writeback_local_only=args.sanplm_writeback_local_only,
        )
        
        # 创建模型
        print("Creating model...")
        model_class = load_model_class(args.model_file)
        print(f"Model source: {args.model_file if args.model_file else 'Rebuilt/BALM_MedualTime.py'}")
        model = model_class(config).float().to(device)
        maybe_apply_sanplm_train_stage(model, args, epoch=0, logger=logger, force=True)

        # 可选：加载预训练权重（支持 strict=False 兼容新增模块）
        if args.pretrained_ckpt:
            ckpt_path = args.pretrained_ckpt
            if os.path.exists(ckpt_path):
                state = torch.load(ckpt_path, map_location=device)
                model_state = model.state_dict()
                filtered_state = {}
                skipped_shape_mismatch = []
                for k, v in state.items():
                    if k not in model_state:
                        continue
                    if model_state[k].shape != v.shape:
                        skipped_shape_mismatch.append(
                            (k, tuple(v.shape), tuple(model_state[k].shape))
                        )
                        continue
                    filtered_state[k] = v
                missing, unexpected = model.load_state_dict(filtered_state, strict=False)
                print(f"Loaded pretrained checkpoint: {ckpt_path}")
                if skipped_shape_mismatch:
                    print(f"  Skipped shape-mismatch keys: {len(skipped_shape_mismatch)}")
                    for key, src_shape, tgt_shape in skipped_shape_mismatch[:8]:
                        print(f"    - {key}: ckpt{src_shape} -> model{tgt_shape}")
                    if len(skipped_shape_mismatch) > 8:
                        print("    ...")
                if missing:
                    print(f"  Missing keys: {len(missing)}")
                if unexpected:
                    print(f"  Unexpected keys: {len(unexpected)}")
            else:
                print(f"Warning: pretrained checkpoint not found: {ckpt_path}")

        finetune_modes = [
            ("channel_mixer_only", args.finetune_channel_mixer_only),
            ("adapter_only", args.finetune_adapter_only),
            ("readout_only", args.finetune_readout_only),
            ("bridge_only", args.finetune_bridge_only),
        ]
        active_finetune_modes = [name for name, enabled in finetune_modes if enabled]
        if len(active_finetune_modes) > 1:
            raise ValueError(f"Only one finetune mode can be enabled, got: {active_finetune_modes}")

        # 可选：仅训练部分参数
        if args.finetune_channel_mixer_only:
            if getattr(model, "use_channel_mixer", False):
                for name, param in model.named_parameters():
                    param.requires_grad = ("channel_mixer" in name)
                print("Enable finetune_channel_mixer_only: only channel_mixer params are trainable.")
            else:
                print("Warning: channel_mixer is disabled; ignore --finetune_channel_mixer_only.")
        elif args.finetune_adapter_only:
            adapter_keywords = (
                "adapter_query_ts",
                "adapter_query_text",
                ".attn.gate",
            )
            for name, param in model.named_parameters():
                param.requires_grad = any(keyword in name for keyword in adapter_keywords)
            print("Enable finetune_adapter_only: only dual-adapter query/gate params are trainable.")
        elif args.finetune_readout_only:
            readout_keywords = (
                "ts_output_refiner",
                "pred_head",
                "normalize",
            )
            for name, param in model.named_parameters():
                param.requires_grad = any(keyword in name for keyword in readout_keywords)
            print("Enable finetune_readout_only: only TS-only readout params are trainable.")
        elif args.finetune_bridge_only:
            bridge_keywords = (
                "adapter_query_ts",
                "adapter_query_text",
                ".attn.gate",
                "calf_q_proj",
                "calf_k_proj",
                "calf_v_proj",
                "cib_encoder",
                "cib_mu",
                "cib_logvar",
                "cib_gate_proj",
                "cib_predictor",
                "cib_z_to_feat",
                "cross_space_fusion",
                "ts_encoder",
                "ts_proj_norm",
                "residual_calibrator",
            )
            if args.bridge_include_pos_norm:
                bridge_keywords = bridge_keywords + (
                    "wpe",
                    ".ln_1",
                    ".ln_2",
                    "ln_f",
                    "ln_f_text",
                    "ln_f_ts",
                )
            if args.bridge_include_patch_head:
                bridge_keywords = bridge_keywords + (
                    "ts_patch_layer",
                    "trend_projection",
                    "seasonal_projection",
                    "output_projection",
                    "output_projection_semantic",
                    "output_projection_skip",
                    "normalize",
                )
            for name, param in model.named_parameters():
                param.requires_grad = any(keyword in name for keyword in bridge_keywords)
            print("Enable finetune_bridge_only: only semantic bridge params are trainable.")
            if args.bridge_include_pos_norm:
                print("  Extra trainable modules: GPT positional embedding and LayerNorm parameters.")
            if args.bridge_include_patch_head:
                print("  Extra trainable modules: patch input layer, forecast head and RevIN parameters.")

        if args.eval_only:
            print("Eval-only mode: skip training and run test directly.")
            criterion_mse = nn.MSELoss()
            if args.eval_progress_override is not None:
                eval_progress = float(np.clip(args.eval_progress_override, 0.0, 1.0))
            else:
                eval_progress = load_checkpoint_progress(ckpt_path, default=1.0) if args.pretrained_ckpt else 1.0
            test_loss, test_preds, test_trues = validate(
                model,
                test_loader,
                criterion_mse,
                device,
                args,
                args.use_pregenerated,
                alignment_progress=eval_progress,
            )
            mae, mse, rmse, mape, mspe = metric(test_preds, test_trues)
            print(f"\n{'='*50}")
            print("Eval-Only Test Results:")
            print(f"  MSE: {mse:.7f}")
            print(f"  MAE: {mae:.7f}")
            print(f"  RMSE: {rmse:.7f}")
            print(f"  MAPE: {mape:.7f}")
            print(f"{'='*50}")
            result_file = os.path.join(path, f"result_s{args.seed}.txt")
            with open(result_file, "w") as f:
                f.write(f"Setting: {setting}\n")
                f.write(f"MSE: {mse:.6f}\n")
                f.write(f"MAE: {mae:.6f}\n")
                f.write(f"RMSE: {rmse:.6f}\n")
                f.write(f"MAPE: {mape:.6f}\n")
            if args.save_preds_npz:
                np.savez_compressed(
                    os.path.join(path, f"preds_s{args.seed}.npz"),
                    preds=test_preds,
                    trues=test_trues,
                )
            analysis_export_path = maybe_export_analysis(
                model,
                train_loader,
                vali_loader,
                test_loader,
                device,
                args,
                args.use_pregenerated,
                eval_progress,
                path,
                setting,
            )
            if analysis_export_path is not None:
                logger.info(f"Analysis export saved: {analysis_export_path}")
            logger.info(f"Final Test - MSE: {mse:.6f}, MAE: {mae:.6f}")
            print(f"Results saved to {result_file}")
            continue
        
        # 统计可训练参数
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
        if active_finetune_modes:
            trainable_param_names = [name for name, p in model.named_parameters() if p.requires_grad]
            print(f"Trainable tensors ({len(trainable_param_names)}):")
            for name in trainable_param_names[:20]:
                print(f"  - {name}")
            if len(trainable_param_names) > 20:
                print("  ...")

        # 优化器和学习率调度器
        if is_sanplm_minimal:
            trainable_params_list = list(model.parameters())
        else:
            trainable_params_list = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(
            trainable_params_list,
            lr=args.learning_rate,
            weight_decay=float(args.weight_decay),
        )
        ema_helper = EMAHelper(model, args.ema_decay) if float(args.ema_decay) > 0.0 else None
        if ema_helper is not None:
            print(f"EMA enabled: decay={args.ema_decay}")
            logger.info(f"EMA enabled: decay={args.ema_decay}")
        
        if args.lradj == "COS":
            cos_tmax = args.train_epochs if int(args.cos_tmax) <= 0 else int(args.cos_tmax)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, cos_tmax), eta_min=1e-8
            )
        else:
            scheduler = lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                steps_per_epoch=len(train_loader),
                pct_start=args.pct_start,
                epochs=args.train_epochs,
                max_lr=args.learning_rate,
            )
        
        # 损失函数：默认纯MSE，可选mix模式
        criterion_l1 = nn.L1Loss()
        criterion_mse = nn.MSELoss()
        
        # 早停
        checkpoint_by_early_stop = args.checkpoint_metric == "early_stop"
        early_stopping = EarlyStopping(
            patience=args.patience,
            verbose=True,
            save_mode=checkpoint_by_early_stop,
        )
        best_checkpoint_metric = None
        checkpoint_metric_name = None
        best_checkpoint_progress = 1.0
        checkpoint_policy = "early-stop metric" if checkpoint_by_early_stop else args.checkpoint_metric
        print(f"Checkpoint policy: {checkpoint_policy}")
        
        # 训练循环
        print("Starting training...")
        time_now = time.time()
        
        for epoch in range(args.train_epochs):
            epoch_time = time.time()
            maybe_apply_sanplm_train_stage(model, args, epoch=epoch, logger=logger)
            target_model = model.module if hasattr(model, "module") else model
            current_progress = float(epoch) / float(max(args.train_epochs - 1, 1))
            current_gumbel_tau = None
            if hasattr(target_model, "set_gumbel_tau_progress"):
                current_gumbel_tau = float(target_model.set_gumbel_tau_progress(current_progress))
            
            # 训练
            train_loss, align_loss, probe_loss, train_l1, train_mse, l1_weight, mse_weight = train_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                criterion_l1,
                criterion_mse,
                device,
                args,
                epoch,
                ema_helper,
                args.use_pregenerated,
            )

            if ema_helper is not None:
                ema_helper.apply_shadow(model)

            current_progress = float(epoch + 1) / float(max(args.train_epochs, 1))
            monitor_key = f"vali_{args.early_stop_metric}"
            progress_bundle, progress_candidates = select_progress_metrics(
                model,
                vali_loader,
                criterion_mse,
                device,
                args,
                args.use_pregenerated,
                current_progress,
                monitor_key,
            )
            selected_progress, monitor_value, vali_loss, vali_mae, vali_mse = progress_bundle
            test_loss, test_preds, test_trues = validate(
                model,
                test_loader,
                criterion_mse,
                device,
                args,
                args.use_pregenerated,
                alignment_progress=selected_progress,
            )
            train_mae = train_l1
            test_mae, test_mse, _, _, _ = metric(test_preds, test_trues)

            epoch_message = (
                f"Epoch: {epoch+1} | "
                f"Train MSE/MAE: {train_mse:.7f}/{train_mae:.7f} | "
                f"Test MSE/MAE: {test_mse:.7f}/{test_mae:.7f} | "
                f"Val MSE/MAE: {vali_mse:.7f}/{vali_mae:.7f} | "
                f"Train Loss: {train_loss:.7f} | Align Loss: {align_loss:.7f} | Probe Loss: {probe_loss:.7f} | "
                f"Val Loss: {vali_loss:.7f} | Test Loss: {test_loss:.7f} | "
                f"Eval Progress: {selected_progress:.2f} | "
                f"Loss Weights(MAE/MSE): {l1_weight:.2f}/{mse_weight:.2f} | "
                f"Cost time: {time.time() - epoch_time:.2f}s"
            )
            if len(progress_candidates) > 1:
                epoch_message += f" | Progress Candidates: {','.join(f'{item:.2f}' for item in progress_candidates)}"
            if current_gumbel_tau is not None:
                epoch_message += f" | Gumbel Tau: {current_gumbel_tau:.4f}"
            print(epoch_message)
            logger.info(epoch_message)
            
            # 早停检查
            metric_pool = {
                "vali_mse": vali_mse,
                "vali_mae": vali_mae,
                "test_mse": test_mse,
                "test_mae": test_mae,
            }
            checkpoint_key = monitor_key if checkpoint_by_early_stop else args.checkpoint_metric
            checkpoint_value = metric_pool[checkpoint_key]

            if not checkpoint_by_early_stop:
                if best_checkpoint_metric is None or checkpoint_value < best_checkpoint_metric:
                    best_checkpoint_metric = checkpoint_value
                    checkpoint_metric_name = checkpoint_key
                    torch.save(model.state_dict(), os.path.join(path, "checkpoint"))
                    best_checkpoint_progress = selected_progress
                    save_checkpoint_progress(os.path.join(path, "checkpoint"), selected_progress)
                    print(f"Checkpoint metric improved ({checkpoint_key}: {checkpoint_value:.7f}). Saving model ...")
                    logger.info(
                        f"Checkpoint saved by {checkpoint_key}: {checkpoint_value:.6f} at epoch {epoch+1} | progress={selected_progress:.2f}"
                    )
            else:
                if best_checkpoint_metric is None or monitor_value < best_checkpoint_metric:
                    best_checkpoint_metric = monitor_value
                    checkpoint_metric_name = monitor_key
                    best_checkpoint_progress = selected_progress
                    save_checkpoint_progress(os.path.join(path, "checkpoint"), selected_progress)

            early_stopping(monitor_value, model, path)
            if early_stopping.early_stop:
                if ema_helper is not None:
                    ema_helper.restore(model)
                print("Early stopping")
                break

            if ema_helper is not None:
                ema_helper.restore(model)
            
            # 学习率调整
            if args.lradj == "COS":
                scheduler.step()
        
        if checkpoint_metric_name is not None:
            logger.info(f"Final checkpoint selected by {checkpoint_metric_name}: {best_checkpoint_metric:.6f}")

        # 加载最佳模型进行最终测试
        best_model_path = os.path.join(path, "checkpoint")
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
        checkpoint_progress = load_checkpoint_progress(best_model_path, default=best_checkpoint_progress)
        
        # 最终测试
        test_loss, test_preds, test_trues = validate(
            model,
            test_loader,
            criterion_mse,
            device,
            args,
            args.use_pregenerated,
            alignment_progress=checkpoint_progress,
        )
        
        # 计算详细指标
        mae, mse, rmse, mape, mspe = metric(test_preds, test_trues)
        
        print(f"\n{'='*50}")
        print(f"Final Test Results:")
        print(f"  MSE: {mse:.7f}")
        print(f"  MAE: {mae:.7f}")
        print(f"  RMSE: {rmse:.7f}")
        print(f"  MAPE: {mape:.7f}")
        print(f"{'='*50}")
        
        # 保存结果
        result_file = os.path.join(path, f"result_s{args.seed}.txt")
        with open(result_file, "w") as f:
            f.write(f"Setting: {setting}\n")
            f.write(f"MSE: {mse:.6f}\n")
            f.write(f"MAE: {mae:.6f}\n")
            f.write(f"RMSE: {rmse:.6f}\n")
            f.write(f"MAPE: {mape:.6f}\n")
        if args.save_preds_npz:
            np.savez_compressed(
                os.path.join(path, f"preds_s{args.seed}.npz"),
                preds=test_preds,
                trues=test_trues,
            )
        analysis_export_path = maybe_export_analysis(
            model,
            train_loader,
            vali_loader,
            test_loader,
            device,
            args,
            args.use_pregenerated,
            checkpoint_progress,
            path,
            setting,
        )
        if analysis_export_path is not None:
            logger.info(f"Analysis export saved: {analysis_export_path}")
        
        # 记录到日志
        logger.info(f"Final Test - MSE: {mse:.6f}, MAE: {mae:.6f}")
        
        print(f"Results saved to {result_file}")
        print(f"Total time: {time.time() - time_now:.2f}s")


if __name__ == "__main__":
    main()
