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
- `hybrid.py` —— Prefill+Decode并发测试脚本

---

### 🏃‍♂️ 性能测试快速上手

**1. SM 缩放性能测试**

运行不同 SM 数量下的 Prefill 性能测试（默认 `q_len=1`，模拟 GQA 场景）：
   ```bash
   python quick_sm_test.py
   ```

**2. Prefill + Decode 并发测试 (hybrid.py)**

这个脚本用于模拟真实推理场景：一个计算密集的Prefill任务和一个延迟敏感的Decode任务并发执行。

#### 核心功能

- **并发与顺序模式**: 支持 `concurrent` (多线程并发) 和 `sequential` (单线程顺序) 两种模式。
- **资源隔离**: Prefill在默认Context下运行，Decode在受限的Green Context下运行。

#### 运行示例

- **运行并发测试 (推荐)**:
  使用30个SM为Decode任务创建Green Context。
  ```bash
  python hybrid.py --sm 30
  ```

- **运行顺序测试 (作为基线)**:
  所有任务都在默认Context下顺序执行，用于对比。
  ```bash
  python hybrid.py --mode sequential
  ```

- **自定义并发测试**:
  使用20个SM，batch size为64，prefill运行5次。
  ```bash
  python hybrid.py --sm 20 --batch_size 64 --runs 5
  ```

#### 如何分析结果

通过对比 `concurrent` 和 `sequential` 模式的总时间，可以清晰地看到并发执行带来的性能收益。如果 `并发总时间` < `顺序执行总时间`，说明并发执行有效。
