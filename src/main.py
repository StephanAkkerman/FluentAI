import os

print(os.environ["CUDA_PATH"])
os.add_dll_directory(os.path.join(os.environ["CUDA_PATH"], "bin"))
