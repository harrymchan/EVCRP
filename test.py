import tensorflow as tf
import timeit

# 检查 TensorFlow 版本
print("TensorFlow Version:", tf.__version__)

# 检查 GPU 设备
print("GPU Device Name:", tf.test.gpu_device_name())
print("Available GPUs:", tf.config.list_physical_devices('GPU'))
print("Available CPUs:", tf.config.list_physical_devices('CPU'))
print("TensorFlow is using GPU:", len(tf.config.list_physical_devices('GPU')) > 0)

# 输出可用的 GPU 数量
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# CPU 计算
def cpu_run():
    with tf.device('/CPU:0'):
        cpu_a = tf.random.normal([10000, 1000])
        cpu_b = tf.random.normal([1000, 2000])
        c = tf.matmul(cpu_a, cpu_b)
    return c

# GPU 计算（如果有 GPU）
def gpu_run():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        with tf.device('/GPU:0'):
            gpu_a = tf.random.normal([10000, 1000])
            gpu_b = tf.random.normal([1000, 2000])
            c = tf.matmul(gpu_a, gpu_b)
        return c
    else:
        print("No GPU available, skipping GPU test.")
        return None

# 计算 CPU 和 GPU 运行时间
cpu_time = timeit.timeit(cpu_run, number=10)
print("CPU time:", cpu_time)

if tf.config.list_physical_devices('GPU'):
    gpu_time = timeit.timeit(gpu_run, number=10)
    print("GPU time:", gpu_time)
else:
    gpu_time = None

# 如果 GPU 存在，比较 CPU 和 GPU 速度
if gpu_time:
    print(f"GPU speedup factor: {cpu_time / gpu_time:.2f}x")

import tensorflow as tf

# 检查 TensorFlow 版本
print("TensorFlow Version:", tf.__version__)

# 检查可用 GPU
gpus = tf.config.list_physical_devices('GPU')
print("Available GPUs:", gpus)

# 强制 TensorFlow 在 GPU 上运行
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)  # 防止 TensorFlow 占满显存
        with tf.device('/GPU:0'):
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            c = tf.matmul(a, b)
        print("GPU computation successful")
    except RuntimeError as e:
        print("GPU Error:", e)
else:
    print("No GPU detected!")
