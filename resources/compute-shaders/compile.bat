@echo OFF

Rem This line sets OS to either 32bit or 64bit depending on the current session
reg Query "HKLM\Hardware\Description\System\CentralProcessor\0" | find /i "x86" > NUL && set OS=32BIT || set OS=64BIT

:: Use 32bit version
if %OS%==32BIT (
    %VK_SDK_PATH%\Bin32\glslc.exe shader.vert -o vert.spv
    %VK_SDK_PATH%\Bin32\glslc.exe shader.frag -o frag.spv
    %VK_SDK_PATH%\Bin32\glslc.exe shader.comp -o compute.spv
) else if %OS%==64BIT (
    %VK_SDK_PATH%\Bin\glslc.exe shader.vert -o vert.spv
    %VK_SDK_PATH%\Bin\glslc.exe shader.frag -o frag.spv
    %VK_SDK_PATH%\Bin\glslc.exe shader.comp -o compute.spv
)
pause