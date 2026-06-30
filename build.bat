@echo off
set NPACK_SDK=C:\Users\mahip\Downloads\npcap-sdk-1.16
set ONNX_INC=onnxruntime\build\native\include
set ONNX_LIB=onnxruntime\runtimes\win-x64\native

g++ real_time_ids.cpp feature_extraction.cpp onnx_model.cpp ^
    -o real_time_ids.exe ^
    -I"%NPACK_SDK%\Include" ^
    -I"%ONNX_INC%" ^
    -L"%ONNX_LIB%" ^
    -L"%NPACK_SDK%\Lib\x64" ^
    -lonnxruntime -lwpcap -static -lws2_32 ^
    -std=c++17 -O2

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Build successful! real_time_ids.exe created.
    echo Make sure onnxruntime.dll is in the same directory as real_time_ids.exe
    copy /Y "%ONNX_LIB%\onnxruntime.dll" . >nul
) else (
    echo.
    echo Build failed.
)
