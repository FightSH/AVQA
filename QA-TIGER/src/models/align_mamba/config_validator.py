"""
对齐配置验证模块

提供对齐配置参数的验证功能，确保配置的有效性和一致性。
"""

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


def validate_alignment_config(config: Dict[str, Any]) -> bool:
    """
    验证对齐配置的有效性
    
    Args:
        config: 包含对齐配置的字典
        
    Returns:
        bool: 配置是否有效
        
    Raises:
        ValueError: 当配置无效时抛出异常
    """
    model_config = config.get('hyper_params', {}).get('model', {})
    
    # 检查是否启用对齐
    use_alignment = model_config.get('use_alignment', False)
    alignment_config = model_config.get('alignment_config', {})
    
    if not use_alignment:
        # 如果未启用对齐，不需要进一步验证
        return True
    
    # 验证对齐配置存在
    if not alignment_config:
        raise ValueError("启用对齐功能时必须提供alignment_config配置")
    
    # 验证enabled字段与use_alignment一致性
    enabled = alignment_config.get('enabled', False)
    if use_alignment != enabled:
        logger.warning(f"use_alignment ({use_alignment}) 与 alignment_config.enabled ({enabled}) 不一致，将使用use_alignment的值")
        alignment_config['enabled'] = use_alignment
    
    # 验证对齐策略
    valid_strategies = ['standard', 'reverse', 'bidirectional']
    strategy = alignment_config.get('strategy', 'standard')
    if strategy not in valid_strategies:
        raise ValueError(f"无效的对齐策略: {strategy}. 有效选项: {valid_strategies}")
    
    # 验证lambda_align范围
    lambda_align = alignment_config.get('lambda_align', 0.1)
    if not isinstance(lambda_align, (int, float)) or not 0 <= lambda_align <= 1:
        raise ValueError(f"lambda_align必须是[0, 1]范围内的数值，当前值: {lambda_align}")
    
    # 验证ot_eps
    ot_eps = alignment_config.get('ot_eps', 1e-8)
    if not isinstance(ot_eps, (int, float)) or ot_eps <= 0:
        raise ValueError(f"ot_eps必须是大于0的数值，当前值: {ot_eps}")
    
    # 验证mmd_sigma
    mmd_sigma = alignment_config.get('mmd_sigma', 1.0)
    if not isinstance(mmd_sigma, (int, float)) or mmd_sigma <= 0:
        raise ValueError(f"mmd_sigma必须是大于0的数值，当前值: {mmd_sigma}")
    
    # 验证布尔类型参数
    bool_params = ['patch_alignment', 'debug_mode', 'memory_efficient']
    for param in bool_params:
        value = alignment_config.get(param, True)
        if not isinstance(value, bool):
            raise ValueError(f"{param}必须是布尔值，当前值: {value} (类型: {type(value)})")
    
    logger.info("对齐配置验证通过")
    return True


def get_default_alignment_config() -> Dict[str, Any]:
    """
    获取默认的对齐配置
    
    Returns:
        Dict[str, Any]: 默认配置字典
    """
    return {
        'enabled': False,
        'strategy': 'standard',
        'lambda_align': 0.1,
        'ot_eps': 1e-8,
        'mmd_sigma': 1.0,
        'patch_alignment': True,
        'debug_mode': False,
        'memory_efficient': True,
    }


def merge_alignment_config(user_config: Dict[str, Any], 
                          default_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    合并用户配置和默认配置
    
    Args:
        user_config: 用户提供的配置
        default_config: 默认配置，如果为None则使用get_default_alignment_config()
        
    Returns:
        Dict[str, Any]: 合并后的配置
    """
    if default_config is None:
        default_config = get_default_alignment_config()
    
    merged_config = default_config.copy()
    merged_config.update(user_config)
    
    return merged_config


def create_alignment_config_template() -> Dict[str, Any]:
    """
    创建对齐配置模板，用于生成示例配置文件
    
    Returns:
        Dict[str, Any]: 配置模板
    """
    return {
        'hyper_params': {
            'model': {
                # 其他模型配置...
                
                # 对齐配置
                'use_alignment': False,  # 是否启用跨模态对齐功能
                'alignment_config': {
                    'enabled': False,  # 是否启用对齐（与use_alignment保持一致）
                    'strategy': 'standard',  # 对齐策略: 'standard', 'reverse', 'bidirectional'
                    'lambda_align': 0.1,  # 对齐损失权重
                    'ot_eps': 1e-8,  # 最优传输数值稳定性参数
                    'mmd_sigma': 1.0,  # MMD高斯核参数
                    'patch_alignment': True,  # 是否对齐patch特征
                    'debug_mode': False,  # 是否启用调试模式
                    'memory_efficient': True,  # 是否使用内存优化模式
                }
            }
        }
    }


def print_alignment_config_help():
    """
    打印对齐配置的帮助信息
    """
    help_text = """
对齐配置参数说明：

1. use_alignment (bool): 是否启用跨模态对齐功能
   - 默认值: False
   - 说明: 主开关，控制是否使用对齐功能

2. alignment_config.enabled (bool): 是否启用对齐
   - 默认值: False
   - 说明: 应与use_alignment保持一致

3. alignment_config.strategy (str): 对齐策略
   - 默认值: 'standard'
   - 可选值: 'standard', 'reverse', 'bidirectional'
   - 说明:
     * standard: 以语言为锚点进行对齐
     * reverse: 以视频为锚点进行对齐（适用于视频问答）
     * bidirectional: 双向对齐，自动选择最优方向

4. alignment_config.lambda_align (float): 对齐损失权重
   - 默认值: 0.1
   - 范围: [0, 1]
   - 说明: 控制对齐损失在总损失中的权重

5. alignment_config.ot_eps (float): 最优传输数值稳定性参数
   - 默认值: 1e-8
   - 范围: > 0
   - 说明: 防止数值计算中的除零错误

6. alignment_config.mmd_sigma (float): MMD高斯核参数
   - 默认值: 1.0
   - 范围: > 0
   - 说明: 控制MMD距离计算中高斯核的带宽

7. alignment_config.patch_alignment (bool): 是否对齐patch特征
   - 默认值: True
   - 说明: 是否对patch特征进行对齐处理

8. alignment_config.debug_mode (bool): 是否启用调试模式
   - 默认值: False
   - 说明: 启用后会返回详细的调试信息

9. alignment_config.memory_efficient (bool): 是否使用内存优化模式
   - 默认值: True
   - 说明: 对于大序列使用分块处理以节省内存

使用建议：
- 对于视频问答任务，建议使用 strategy='reverse'
- 如果内存有限，确保 memory_efficient=True
- 调试时可以启用 debug_mode=True 获取详细信息
- lambda_align 建议从 0.1 开始调优
"""
    print(help_text)


if __name__ == "__main__":
    # 示例用法
    print("=== 对齐配置验证示例 ===")
    
    # 创建示例配置
    example_config = create_alignment_config_template()
    example_config['hyper_params']['model']['use_alignment'] = True
    example_config['hyper_params']['model']['alignment_config']['enabled'] = True
    
    try:
        validate_alignment_config(example_config)
        print("✓ 配置验证通过")
    except ValueError as e:
        print(f"✗ 配置验证失败: {e}")
    
    print("\n=== 配置帮助信息 ===")
    print_alignment_config_help()