@echo off
rem Build kv-stream llama.cpp native DLLs (classic shared layout, CUDA sm_120)
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
set PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin;%PATH%
set CMAKE=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe
set SRC=%~dp0vendor\llama.cpp
set BLD=%SRC%\build

if "%1"=="configure" (
  "%CMAKE%" -S "%SRC%" -B "%BLD%" -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DBUILD_SHARED_LIBS=ON ^
    -DGGML_CUDA=ON ^
    -DCMAKE_CUDA_ARCHITECTURES=120-real ^
    -DGGML_CUDA_FORCE_MMQ=ON ^
    -DCUDA_SEPARABLE_COMPILATION=ON ^
    -DCMAKE_CUDA_FLAGS=--diag-suppress=177,221,550 ^
    -DGGML_NATIVE=ON ^
    -DGGML_OPENMP=ON ^
    -DLLAMA_BUILD_TESTS=OFF ^
    -DLLAMA_BUILD_EXAMPLES=OFF ^
    -DLLAMA_BUILD_SERVER=OFF ^
    -DLLAMA_CURL=OFF ^
    -DLLAMA_BUILD_TOOLS=ON ^
    -DLLAMA_BUILD_BORINGSSL=OFF
  exit /b %errorlevel%
)

if "%1"=="build" (
  "%CMAKE%" --build "%BLD%" --target llama llama-common mtmd -j 16
  exit /b %errorlevel%
)

echo usage: build-kvstream.bat configure^|build
exit /b 1
