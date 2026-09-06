@echo off
rem Build the kv-stream CUDA wheel locally:  build-wheel.bat [path\to\python.exe]
rem
rem Tested with VS2022 Community (MSVC 14.44) + CUDA Toolkit 13.0 on Windows.
rem Kernels are compiled for sm_120a only (RTX 50 series) — adjust
rem CMAKE_CUDA_ARCHITECTURES for other GPUs. GGML_CUDA_FA_ALL_QUANTS is
rem required for the q8_0/q4_0 KV cache FA paths used by adaptive KV streaming.
rem The wheel does NOT bundle cublas64_13.dll / cublasLt64_13.dll; they are
rem resolved from the CUDA toolkit (bin\x64) at runtime.
setlocal
set "PYTHON=%~1"
if "%PYTHON%"=="" set "PYTHON=python"

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set "PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin"
set "CMAKE_ARGS=-DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=120a"
set "FORCE_CMAKE=1"
cd /d "%~dp0"
"%PYTHON%" -m pip install . --no-cache-dir --disable-pip-version-check
