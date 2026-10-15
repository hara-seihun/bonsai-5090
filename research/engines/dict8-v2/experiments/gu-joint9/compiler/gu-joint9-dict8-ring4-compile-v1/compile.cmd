@echo off
call "C:/work\models\tools\msvc\setup_x64.bat"
cd /d "%~dp0"
"C:/work\models\tools\cuda\bin\nvcc.exe" --cubin -std=c++17 -O3 --expt-relaxed-constexpr --generate-code=arch=compute_120a,code=sm_120a -lineinfo -Xptxas=-v -DH5_PROFILE_GEMM -DH5_PDL_EARLY_GEMM_RELEASE -DH5_PDL_GDN_STATE_OVERLAP -DH5_NO_L2PF -DH5_PROFILE_SCALE_CARRIER -DH5_WEIGHT_TMA_CACHE_POLICY=2 -DNOMINMAX -Isrc -I"C:/work\models\source\llama\include" -I"C:/work\models\source\llama\vendor" -I"C:/work\models\source\llama\ggml\include" src/probe.cu -o probe.cubin
exit /b %ERRORLEVEL%
