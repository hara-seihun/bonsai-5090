@echo off
call "C:/work\models\tools\msvc\setup_x64.bat"
set "CUDA_PATH=C:/work\models\tools\cuda"
set "PATH=C:/work\models\tools\cuda\bin;C:/work\models\tools\ninja;C:/work\models\tools\cmake\cmake-4.1.2-windows-x86_64\bin;%PATH%"
cd /d "%~dp0"
cl /nologo /O2 /EHsc /std:c++17 /DNOMINMAX /D_CRT_SECURE_NO_WARNINGS /Isrc cache_tool.cpp src/gguf.cpp /Fe:cache_tool.exe /link bcrypt.lib
if errorlevel 1 exit /b 1
cl /nologo /O2 /EHsc /std:c++17 /DNOMINMAX /D_CRT_SECURE_NO_WARNINGS /Isrc fault_test.cpp src/gguf.cpp /Fe:fault_test.exe /link bcrypt.lib
if errorlevel 1 exit /b 1
"C:/work\models\tools\python\python.exe" experiments/gu-dict8-cache/test_identity.py
if errorlevel 1 exit /b 1
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=OFF
if errorlevel 1 exit /b 1
cmake --build build --parallel 2
exit /b %ERRORLEVEL%
