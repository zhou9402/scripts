## 🚀 Green Context 库编译指南
### 🛠️ 编译步骤

1. **检查依赖环境**  
   需提前安装 `torch` 和 `pybind11`，可使用如下命令安装：
   ```bash
   pip install torch pybind11
   ```

2. **执行编译脚本**  
   在 `semi-pd` 目录下运行：
   ```bash
   bash build_library.sh
   ```
   编译成功后会生成 `green_context_lib*.so` 文件。

3. **导入库测试**  
   编译完成后可在 Python 中直接导入：
   ```python
   import green_context_lib as green_context
   ```

---

### 📦 目录结构说明

- `build_library.sh` —— 一键编译脚本
- `green_context_lib*.so` —— 编译生成的核心库
- `quick_sm_test.py` —— SM缩放性能测试脚本
- `simple_flashinfer.py` —— FlashInfer测试脚本

---

### 🏃‍♂️ 性能测试快速上手

运行不同 SM 数量下的 Prefill 性能测试（默认 `q_len=1`，模拟 GQA 场景）：
   ```bash
   python quick_sm_test.py
   ```
