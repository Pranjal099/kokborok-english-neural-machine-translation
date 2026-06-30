import time
import os

data = os.urandom(100 * 1024 * 1024)  # 100 MB random data

# Write test
start = time.time()
with open("test_file.bin", "wb") as f:
    f.write(data)
end = time.time()
print("Write Speed:", 100 / (end - start), "MB/s")

# Read test
start = time.time()
with open("test_file.bin", "rb") as f:
    f.read()
end = time.time()
print("Read Speed:", 100 / (end - start), "MB/s")