=== 编译 Green Context 库 ===
./build_library.sh 

=== 运行测试，不同SM count下的性能，默认q_len为1，模拟GQA ===
python quick_sm_test.py