#!/bin/bash

echo "=== 编译 Green Context 库 ==="

# 检查必要的依赖
echo "检查环境..."
python3 -c "import torch; import pybind11; print('✓ 依赖检查通过')" || {
    echo "❌ 缺少必要依赖，请安装:"
    echo "pip install torch pybind11"
    exit 1
}

# 检查 CUDA
if ! command -v nvcc &> /dev/null; then
    echo "❌ 找不到 nvcc，请确保 CUDA 已安装并在 PATH 中"
    exit 1
fi

echo "✓ CUDA 环境检查通过"

# 清理之前的编译文件
echo "清理旧文件..."
rm -rf build/ dist/ *.egg-info/
rm -f *.so

# 编译库
echo "开始编译..."
python3 setup.py build_ext --inplace || {
    echo "❌ 编译失败"
    exit 1
}

echo "✓ 编译成功！"

# 检查生成的库文件
if ls green_context_lib*.so 1> /dev/null 2>&1; then
    echo "✓ 库文件已生成:"
    ls -la green_context_lib*.so
else
    echo "❌ 未找到生成的库文件"
    exit 1
fi

echo ""
echo "=== 编译完成 ==="
echo "现在可以直接导入使用："
echo "import green_context"
echo "" 