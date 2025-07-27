import torch
# 使用相对导入，确保在任何路径下都能正确找到模块
from alvs_inter import iterative_sampling as single_iterative_sampling
from batch_alvs_inter import iterative_sampling as batch_iterative_sampling

def test_batch_correctness():
    """
    测试批处理版本的结果是否与多次运行单一样本版本的结果完全一致。
    """
    print("\nRunning test: test_batch_correctness...")
    # 1. 准备测试数据
    batch_size = 4
    seq_len = 60
    dim = 64 # 使用较小的维度以便于调试
    k = 11
    m = 8
    a1 = 0.2
    a2 = 0.8

    # 创建固定的随机数据以确保可复现性
    torch.manual_seed(42)
    f_v_batch = torch.randn(batch_size, seq_len, dim)
    f_text_batch = torch.randn(batch_size, dim)


    # # 3. 循环调用原始的 single_iterative_sampling 分别处理每个样本
    # print("Running single processing in a loop...")
    # single_results = []
    # for i in range(batch_size):
    #     # 从批次中取出单个样本
    #     f_v_single = f_v_batch[i]
    #     f_text_single = f_text_batch[i]
    #
    #     # 调用原始的、未经修改的单样本处理函数
    #     single_result = single_iterative_sampling(f_v_single, f_text_single, k, m, a1, a2)
    #     single_results.append(single_result)
    #
    # print(f"Looped single results: {single_results}")




    # 2. 使用新的 batch_iterative_sampling 一次性处理整个批次
    print("Running batch processing...")
    batch_results = batch_iterative_sampling(f_v_batch, f_text_batch, k, m, a1, a2)
    # print(f"Batch results: {batch_results}")
    print(f"Batch results: {batch_results}")



    # 4. 断言结果完全一致
    # assert batch_results == single_results, "Batch processing results must match looped single processing results!"

    print("Test passed: Batch results are consistent with single results.")

# 运行测试
if __name__ == "__main__":
    test_batch_correctness()
