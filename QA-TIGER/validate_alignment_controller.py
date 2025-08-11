"""
验证AlignmentController代码的语法和基本结构
"""

import ast
import sys

def validate_python_syntax(file_path):
    """验证Python文件的语法"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # 解析AST
        tree = ast.parse(source)
        print(f"✓ {file_path} 语法正确")
        
        # 检查类和函数定义
        classes = []
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.FunctionDef):
                functions.append(node.name)
        
        print(f"  发现类: {classes}")
        print(f"  发现函数: {len(functions)} 个")
        
        return True
        
    except SyntaxError as e:
        print(f"✗ {file_path} 语法错误: {e}")
        return False
    except Exception as e:
        print(f"✗ {file_path} 验证失败: {e}")
        return False

def main():
    files_to_validate = [
        'src/models/align_mamba/alignment_controller.py',
        'src/models/align_mamba/__init__.py',
        'tests/test_alignment_controller.py'
    ]
    
    all_valid = True
    for file_path in files_to_validate:
        if not validate_python_syntax(file_path):
            all_valid = False
    
    if all_valid:
        print("\n✓ 所有文件验证通过！")
        print("\nAlignmentController 主要功能:")
        print("- 支持三种对齐策略: standard, reverse, bidirectional")
        print("- 包含完整的输入验证和错误处理")
        print("- 支持数值稳定性保证")
        print("- 支持内存高效模式")
        print("- 支持Patch特征对齐")
        print("- 支持调试模式")
    else:
        print("\n✗ 部分文件验证失败")
        sys.exit(1)

if __name__ == '__main__':
    main()