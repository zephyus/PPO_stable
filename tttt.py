python - << 'EOF'
import sys, utils, inspect

# 1) Python 搜索路径
print("sys.path:", sys.path)

# 2) utils 模块真实位置
print("utils.__file__:", utils.__file__)

# 3) Trainer 类所属模块
print("Trainer.__module__:", utils.Trainer.__module__)

# 额外：查看 Trainer __init__ 签名，帮我们给出正确的实例化参数
print("Trainer.__init__ signature:", inspect.signature(utils.Trainer.__init__))
EOF
