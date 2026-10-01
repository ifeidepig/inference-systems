# Five-Process Stability Summary

Environment: RTX 3060 Laptop, CUDA 13.4, release build. Each process used seven CUDA-Event trials per shape.

| Path | Mean vs cuBLAS | Std. dev. | Min | Max | Dispatch decision |
| --- | ---: | ---: | ---: | ---: | --- |
| FP16 QKV, M=16 N=6144 K=1024 | 1.014x | 0.063 | 0.911x | 1.108x | cuBLAS fallback |
| BF16 QKV, M=16 N=6144 K=1024 | 1.926x | 0.047 | 1.892x | 2.017x | direct WMMA |
| BF16 Gate-Up, M=16 N=7168 K=1024 | 1.724x | 0.011 | 1.714x | 1.742x | direct WMMA |

All 110 dtype/shape rows across the five processes passed correctness validation.
