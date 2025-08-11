"""
AlignMamba模块

该模块提供跨模态对齐功能，包括：
- OptimalTransportAlignment: 基于最优传输的局部对齐
- MMDGlobalAlignment: 基于MMD的全局对齐
- AlignmentController: 对齐控制器，提供统一的对齐接口
- AVQAAlignMamba: 完整的对齐框架（原始实现）
"""

from .qa_tiger_align_mamba import (
    OptimalTransportAlignment,
    MMDGlobalAlignment,
    AVQAAlignMamba
)

from .alignment_controller import AlignmentController
from .config_validator import (
    validate_alignment_config,
    get_default_alignment_config,
    merge_alignment_config,
    create_alignment_config_template,
    print_alignment_config_help
)

__all__ = [
    'OptimalTransportAlignment',
    'MMDGlobalAlignment', 
    'AVQAAlignMamba',
    'AlignmentController',
    'validate_alignment_config',
    'get_default_alignment_config',
    'merge_alignment_config',
    'create_alignment_config_template',
    'print_alignment_config_help'
]