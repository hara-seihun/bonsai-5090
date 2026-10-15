@echo off
call "C:/work\models\tools\msvc\setup_x64.bat"
cd /d "%~dp0"
dumpbin /exports build\quality_custom.dll
dumpbin /imports cache_tool.exe
dumpbin /imports fault_test.exe
