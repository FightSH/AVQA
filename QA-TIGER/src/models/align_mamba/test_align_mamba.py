"""
AlignMamba实现的单元测试

验证各个组件的功能正确性，包括：
1. 最优传输对齐模块测试
2. MMD全局对齐模块测试
3. Mamba骨干网络测试
4. 完整模型测试
5. 数学公式验证
"""

import torch
import torch.nn as nn
import numpy as np
import pytest
from typing import Tuple

from align_mamba_implementation import (
    OptimalTransportAlignment,
    MMDGlobalAlignment,
    MambaBackbone,
    AlignMamba,
    create_align_mamba_classifier,
    create_align_mamba_regressor,
    create_align_mamba_feature_extractor
)


class TestOptimalTransportAlignment:
    """测试最优传输对齐模块"""
    
    def setup_method(self):
        """测试前的设置"""
        self.ot_aligner = OptimalTransportAlignment()
        self.batch_size = 2
        self.src_len = 10
        self.tgt_len = 8
        self.d_model = 64
        
        # 创建测试数据
        self.src_tokens = torch.randn(self.batch_size, self.src_len, self.d_model)
        self.tgt_tokens = torch.randn(self.batch_size, self.tgt_len, self.d_model)
    
    def test_cost_matrix_computation(self):
        """测试成本矩阵计算"""
        cost_matrix = self.ot_aligner.compute_cost_matrix(self.src_tokens, self.tgt_tokens)
        
        # 检查形状
        assert cost_matrix.shape == (self.batch_size, self.src_len, self.tgt_len)
        
        # 检查数值范围 (余弦距离应在[0,2]范围内)
        assert torch.all(cost_matrix >= 0)
        assert torch.all(cost_matrix <= 2)
        
        # 检查对称性质：相同向量的成本应为0
        identical_tokens = self.src_tokens[:, :self.tgt_len, :]  # 截取相同长度
        cost_identical = self.ot_aligner.compute_cost_matrix(identical_tokens, identical_tokens)
        diagonal_costs = torch.diagonal(cost_identical, dim1=1, dim2=2)
        assert torch.allclose(diagonal_costs, torch.zeros_like(diagonal_costs), atol=1e-6)
    
    def test_relaxed_ot_solution(self):
        """测试松弛版OT求解"""
        cost_matrix = self.ot_aligner.compute_cost_matrix(self.src_tokens, self.tgt_tokens)
        transport_matrix = self.ot_aligner.solve_relaxed_ot(cost_matrix)
        
        # 检查形状
        assert transport_matrix.shape == cost_matrix.shape
        
        # 检查约束条件：每行和应为1/T_src
        row_sums = transport_matrix.sum(dim=-1)
        expected_sum = 1.0 / self.src_len
        assert torch.allclose(row_sums, torch.full_like(row_sums, expected_sum), atol=1e-6)
        
        # 检查稀疏性：每行应只有一个非零元素
        non_zero_counts = (transport_matrix > 0).sum(dim=-1)
        assert torch.all(non_zero_counts == 1)
        
        # 检查非负性
        assert torch.all(transport_matrix >= 0)
    
    def test_transport_application(self):
        """测试传输矩阵应用"""
        cost_matrix = self.ot_aligner.compute_cost_matrix(self.src_tokens, self.tgt_tokens)
        transport_matrix = self.ot_aligner.solve_relaxed_ot(cost_matrix)
        aligned_features = self.ot_aligner.apply_transport(self.src_tokens, transport_matrix)
        
        # 检查输出形状
        assert aligned_features.shape == (self.batch_size, self.tgt_len, self.d_model)
        
        # 检查数值合理性（不应包含NaN或Inf）
        assert torch.all(torch.isfinite(aligned_features))
    
    def test_forward_pass(self):
        """测试完整前向传播"""
        aligned_features, transport_matrix = self.ot_aligner(self.src_tokens, self.tgt_tokens)
        
        assert aligned_features.shape == (self.batch_size, self.tgt_len, self.d_model)
        assert transport_matrix.shape == (self.batch_size, self.src_len, self.tgt_len)
        assert torch.all(torch.isfinite(aligned_features))


class TestMMDGlobalAlignment:
    """测试MMD全局对齐模块"""
    
    def setup_method(self):
        """测试前的设置"""
        self.mmd_aligner = MMDGlobalAlignment(sigma=1.0)
        self.batch_size = 2
        self.seq_len = 10
        self.d_model = 64
        
        # 创建测试数据
        self.x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        self.y = torch.randn(self.batch_size, self.seq_len, self.d_model)
        self.z = torch.randn(self.batch_size, self.seq_len, self.d_model)
    
    def test_gaussian_kernel(self):
        """测试高斯核计算"""
        kernel_matrix = self.mmd_aligner.gaussian_kernel(self.x, self.y)
        
        # 检查形状
        assert kernel_matrix.shape == (self.batch_size, self.seq_len, self.seq_len)
        
        # 检查数值范围 (高斯核值应在[0,1]范围内)
        assert torch.all(kernel_matrix >= 0)
        assert torch.all(kernel_matrix <= 1)
        
        # 检查对称性：K(x,x)的对角线应为1
        kernel_xx = self.mmd_aligner.gaussian_kernel(self.x, self.x)
        diagonal = torch.diagonal(kernel_xx, dim1=1, dim2=2)
        assert torch.allclose(diagonal, torch.ones_like(diagonal), atol=1e-6)
    
    def test_mmd_computation(self):
        """测试MMD计算"""
        mmd_squared = self.mmd_aligner.compute_mmd_squared(self.x, self.y)
        
        # 检查形状
        assert mmd_squared.shape == (self.batch_size,)
        
        # 检查非负性
        assert torch.all(mmd_squared >= 0)
        
        # 检查相同分布的MMD应接近0
        mmd_self = self.mmd_aligner.compute_mmd_squared(self.x, self.x)
        assert torch.allclose(mmd_self, torch.zeros_like(mmd_self), atol=1e-5)
    
    def test_global_alignment_loss(self):
        """测试全局对齐损失"""
        loss = self.mmd_aligner(self.x, self.y, self.z)
        
        # 检查输出为标量
        assert loss.dim() == 0
        
        # 检查非负性
        assert loss >= 0
        
        # 检查数值合理性
        assert torch.isfinite(loss)


class TestMambaBackbone:
    """测试Mamba骨干网络"""
    
    def setup_method(self):
        """测试前的设置"""
        self.d_model = 128
        self.n_layers = 2
        self.backbone = MambaBackbone(
            d_model=self.d_model,
            n_layers=self.n_layers,
            dropout=0.1
        )
        
        self.batch_size = 2
        self.seq_len = 20
        self.input_seq = torch.randn(self.batch_size, self.seq_len, self.d_model)
    
    def test_forward_pass(self):
        """测试前向传播"""
        output = self.backbone(self.input_seq)
        
        # 检查输出形状
        assert output.shape == self.input_seq.shape
        
        # 检查数值合理性
        assert torch.all(torch.isfinite(output))
    
    def test_gradient_flow(self):
        """测试梯度流"""
        output = self.backbone(self.input_seq)
        loss = output.sum()
        loss.backward()
        
        # 检查梯度是否正确计算
        for param in self.backbone.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert torch.all(torch.isfinite(param.grad))


class TestAlignMamba:
    """测试完整的AlignMamba模型"""
    
    def setup_method(self):
        """测试前的设置"""
        self.dim_audio = 128
        self.dim_video = 256
        self.dim_language = 768
        self.d_model = 64
        self.num_classes = 5
        
        self.batch_size = 2
        self.audio_len = 15
        self.video_len = 12
        self.language_len = 10
        
        # 创建测试数据
        self.audio = torch.randn(self.batch_size, self.audio_len, self.dim_audio)
        self.video = torch.randn(self.batch_size, self.video_len, self.dim_video)
        self.language = torch.randn(self.batch_size, self.language_len, self.dim_language)
        self.labels = torch.randint(0, self.num_classes, (self.batch_size,))
    
    def test_classifier_creation(self):
        """测试分类器创建"""
        model = create_align_mamba_classifier(
            dim_audio=self.dim_audio,
            dim_video=self.dim_video,
            dim_language=self.dim_language,
            num_classes=self.num_classes,
            d_model=self.d_model
        )
        
        assert isinstance(model, AlignMamba)
        assert model.task_type == "classification"
    
    def test_regressor_creation(self):
        """测试回归器创建"""
        model = create_align_mamba_regressor(
            dim_audio=self.dim_audio,
            dim_video=self.dim_video,
            dim_language=self.dim_language,
            output_dim=1,
            d_model=self.d_model
        )
        
        assert isinstance(model, AlignMamba)
        assert model.task_type == "regression"
    
    def test_feature_extractor_creation(self):
        """测试特征提取器创建"""
        model = create_align_mamba_feature_extractor(
            dim_audio=self.dim_audio,
            dim_video=self.dim_video,
            dim_language=self.dim_language,
            d_model=self.d_model
        )
        
        assert isinstance(model, AlignMamba)
        assert model.task_type == "feature_extraction"
    
    def test_classification_forward(self):
        """测试分类任务前向传播"""
        model = create_align_mamba_classifier(
            dim_audio=self.dim_audio,
            dim_video=self.dim_video,
            dim_language=self.dim_language,
            num_classes=self.num_classes,
            d_model=self.d_model
        )
        
        outputs = model(self.audio, self.video, self.language, labels=self.labels)
        
        # 检查输出键
        assert "logits" in outputs
        assert "loss" in outputs
        assert "task_loss" in outputs
        assert "alignment_loss" in outputs
        
        # 检查形状
        assert outputs["logits"].shape == (self.batch_size, self.num_classes)
        
        # 检查损失
        assert outputs["loss"].dim() == 0
        assert outputs["task_loss"].dim() == 0
        assert outputs["alignment_loss"].dim() == 0
        
        # 检查数值合理性
        assert torch.all(torch.isfinite(outputs["logits"]))
        assert torch.isfinite(outputs["loss"])
    
    def test_regression_forward(self):
        """测试回归任务前向传播"""
        model = create_align_mamba_regressor(
            dim_audio=self.dim_audio,
            dim_video=self.dim_video,
            dim_language=self.dim_language,
            output_dim=1,
            d_model=self.d_model
        )
        
        reg_labels = torch.randn(self.batch_size, 1)
        outputs = model(self.audio, self.video, self.language, labels=reg_labels)
        
        # 检查输出
        assert "logits" in outputs
        assert outputs["logits"].shape == (self.batch_size, 1)
        assert torch.isfinite(outputs["loss"])
    
    def test_feature_extraction_forward(self):
        """测试特征提取前向传播"""
        model = create_align_mamba_feature_extractor(
            dim_audio=self.dim_audio,
            dim_video=self.dim_video,
            dim_language=self.dim_language,
            d_model=self.d_model
        )
        
        outputs = model(
            self.audio, self.video, self.language,
            return_features=True, return_alignments=True
        )
        
        # 检查输出键
        assert "features" in outputs
        assert "alignment_loss" in outputs
        assert "transport_matrices" in outputs
        assert "aligned_features" in outputs
        
        # 检查特征形状
        expected_seq_len = self.language_len * 3  # 交错后的序列长度
        assert outputs["features"].shape == (self.batch_size, expected_seq_len, self.d_model)
    
    def test_interleaved_sequence_construction(self):
        """测试交错序列构建"""
        model = create_align_mamba_feature_extractor(
            dim_audio=self.dim_audio,
            dim_video=self.dim_video,
            dim_language=self.dim_language,
            d_model=self.d_model
        )
        
        # 创建对齐后的特征（都应该有相同的时间长度）
        aligned_audio = torch.randn(self.batch_size, self.language_len, self.d_model)
        aligned_video = torch.randn(self.batch_size, self.language_len, self.d_model)
        language_proj = torch.randn(self.batch_size, self.language_len, self.d_model)
        
        interleaved = model.build_interleaved_sequence(aligned_video, aligned_audio, language_proj)
        
        # 检查形状
        expected_len = self.language_len * 3
        assert interleaved.shape == (self.batch_size, expected_len, self.d_model)
        
        # 检查交错模式：每3个位置应该对应[video, audio, language]
        for t in range(self.language_len):
            base_idx = t * 3
            # 检查是否正确交错（这里只检查形状，实际值可能因为模型参数而不同）
            assert torch.allclose(
                interleaved[:, base_idx, :], aligned_video[:, t, :], atol=1e-6
            )
            assert torch.allclose(
                interleaved[:, base_idx + 1, :], aligned_audio[:, t, :], atol=1e-6
            )
            assert torch.allclose(
                interleaved[:, base_idx + 2, :], language_proj[:, t, :], atol=1e-6
            )
    
    def test_gradient_computation(self):
        """测试梯度计算"""
        model = create_align_mamba_classifier(
            dim_audio=self.dim_audio,
            dim_video=self.dim_video,
            dim_language=self.dim_language,
            num_classes=self.num_classes,
            d_model=self.d_model
        )
        
        outputs = model(self.audio, self.video, self.language, labels=self.labels)
        loss = outputs["loss"]
        loss.backward()
        
        # 检查梯度
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"参数 {name} 没有梯度"
                assert torch.all(torch.isfinite(param.grad)), f"参数 {name} 的梯度包含NaN或Inf"
    
    def test_different_sequence_lengths(self):
        """测试不同序列长度的处理"""
        model = create_align_mamba_classifier(
            dim_audio=self.dim_audio,
            dim_video=self.dim_video,
            dim_language=self.dim_language,
            num_classes=self.num_classes,
            d_model=self.d_model
        )
        
        # 创建不同长度的序列
        audio_short = torch.randn(1, 5, self.dim_audio)
        video_long = torch.randn(1, 25, self.dim_video)
        language_medium = torch.randn(1, 15, self.dim_language)
        labels_single = torch.randint(0, self.num_classes, (1,))
        
        # 应该能正常处理
        outputs = model(audio_short, video_long, language_medium, labels=labels_single)
        
        assert "logits" in outputs
        assert outputs["logits"].shape == (1, self.num_classes)


class TestMathematicalProperties:
    """测试数学性质的正确性"""
    
    def test_ot_transport_conservation(self):
        """测试OT传输的质量守恒性质"""
        ot_aligner = OptimalTransportAlignment()
        
        batch_size, src_len, tgt_len, d_model = 2, 10, 8, 64
        src_tokens = torch.randn(batch_size, src_len, d_model)
        tgt_tokens = torch.randn(batch_size, tgt_len, d_model)
        
        cost_matrix = ot_aligner.compute_cost_matrix(src_tokens, tgt_tokens)
        transport_matrix = ot_aligner.solve_relaxed_ot(cost_matrix)
        
        # 检查行和约束：每行和应为1/src_len
        row_sums = transport_matrix.sum(dim=-1)
        expected_row_sum = 1.0 / src_len
        assert torch.allclose(row_sums, torch.full_like(row_sums, expected_row_sum))
    
    def test_mmd_properties(self):
        """测试MMD的数学性质"""
        mmd_aligner = MMDGlobalAlignment(sigma=1.0)
        
        batch_size, seq_len, d_model = 2, 10, 64
        x = torch.randn(batch_size, seq_len, d_model)
        y = torch.randn(batch_size, seq_len, d_model)
        
        # 性质1: MMD(X,X) = 0
        mmd_self = mmd_aligner.compute_mmd_squared(x, x)
        assert torch.allclose(mmd_self, torch.zeros_like(mmd_self), atol=1e-5)
        
        # 性质2: MMD(X,Y) = MMD(Y,X) (对称性)
        mmd_xy = mmd_aligner.compute_mmd_squared(x, y)
        mmd_yx = mmd_aligner.compute_mmd_squared(y, x)
        assert torch.allclose(mmd_xy, mmd_yx, atol=1e-6)
        
        # 性质3: MMD(X,Y) >= 0 (非负性)
        assert torch.all(mmd_xy >= 0)
    
    def test_cosine_distance_properties(self):
        """测试余弦距离的性质"""
        ot_aligner = OptimalTransportAlignment()
        
        batch_size, seq_len, d_model = 2, 5, 64
        
        # 创建单位向量
        x = torch.randn(batch_size, seq_len, d_model)
        x = x / x.norm(dim=-1, keepdim=True)
        
        # 性质1: 相同向量的余弦距离为0
        cost_self = ot_aligner.compute_cost_matrix(x, x)
        diagonal = torch.diagonal(cost_self, dim1=1, dim2=2)
        assert torch.allclose(diagonal, torch.zeros_like(diagonal), atol=1e-6)
        
        # 性质2: 余弦距离在[0,2]范围内
        y = torch.randn(batch_size, seq_len, d_model)
        y = y / y.norm(dim=-1, keepdim=True)
        cost_matrix = ot_aligner.compute_cost_matrix(x, y)
        assert torch.all(cost_matrix >= 0)
        assert torch.all(cost_matrix <= 2)


def run_all_tests():
    """运行所有测试"""
    print("开始运行AlignMamba测试套件...")
    
    # 创建测试实例
    test_classes = [
        TestOptimalTransportAlignment,
        TestMMDGlobalAlignment,
        TestMambaBackbone,
        TestAlignMamba,
        TestMathematicalProperties
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n运行 {test_class.__name__} 测试...")
        test_instance = test_class()
        
        # 获取所有测试方法
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                # 运行setup方法（如果存在）
                if hasattr(test_instance, 'setup_method'):
                    test_instance.setup_method()
                
                # 运行测试方法
                test_method = getattr(test_instance, method_name)
                test_method()
                
                print(f"  ✓ {method_name}")
                passed_tests += 1
                
            except Exception as e:
                print(f"  ✗ {method_name}: {str(e)}")
    
    print(f"\n测试完成: {passed_tests}/{total_tests} 通过")
    
    if passed_tests == total_tests:
        print("🎉 所有测试通过！")
    else:
        print(f"⚠️  {total_tests - passed_tests} 个测试失败")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)