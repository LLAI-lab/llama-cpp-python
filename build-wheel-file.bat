@echo off
rem Build kv-stream CUDA wheel FILE - same env/flags as build-wheel.bat,
rem but keeps the .whl in dist\ instead of installing into the env.
setlocal
set "PYTHON=python"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set "PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin"
set "CMAKE_ARGS=-DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=120a"
set "FORCE_CMAKE=1"
cd /d "%~dp0"
if not exist dist mkdir dist
"%PYTHON%" -m pip wheel . --no-deps --no-cache-dir --disable-pip-version-check -w dist
exit /b %errorlevel%
