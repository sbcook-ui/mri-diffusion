FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
RUN pip install --no-cache-dir scipy h5py matplotlib