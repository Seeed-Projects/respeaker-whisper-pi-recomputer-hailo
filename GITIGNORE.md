# .gitignore 文件说明

此项目使用 `.gitignore` 文件来忽略不应提交到版本控制的文件和目录。

## 被忽略的文件类型

### 1. Python相关文件
- `__pycache__/` 目录 - Python字节码缓存
- `*.pyc`, `*.pyo`, `*.pyd` - 编译后的Python文件
- `*.so` - 编译后的共享库
- `build/`, `dist/`, `*.egg-info/` - Python打包相关文件

### 2. 虚拟环境
- `venv/`, `env/`, `.venv/`, `whisper_env/` - Python虚拟环境目录

### 3. IDE和编辑器文件
- `.idea/` - PyCharm/IntelliJ配置
- `.vscode/` - Visual Studio Code配置
- `*.swp`, `*.swo` - Vim交换文件
- `.DS_Store` - macOS系统文件

### 4. 日志和临时文件
- `*.log` - 日志文件
- `tmp/`, `temp/` - 临时目录
- `*.tmp`, `*.temp` - 临时文件

### 5. 下载的资源和模型
- `app/hefs/` - Hailo模型文件目录
- `app/decoder_assets/` - 解码器资源文件
- `*.hef` - Hailo执行格式文件
- `*.npy` - NumPy数组文件

### 6. 测试文件
- `test_*.py`, `*_test.py` - 测试Python文件
- `tests/` - 测试目录
- `*_test_*.py` - 测试相关文件

## 注意事项

1. **资源文件**: 模型文件(.hef)和资源文件(.npy)不包含在版本控制中，需要单独下载
2. **虚拟环境**: 开发环境不包含在版本控制中，需要重新创建
3. **配置文件**: 本地配置文件应被忽略，使用模板文件替代

## 使用说明

要初始化开发环境，请运行:
```bash
python3 setup.py
source whisper_env/bin/activate
```

要下载必要的资源文件，请运行:
```bash
python3 ./download_resources.py --hw-arch hailo8
```