# Execute from PowerShell: powershell -ExecutionPolicy Bypass -File .\setup_gpu.ps1
# Python 2.7 is first on PATH on this laptop, so explicitly choose Python 3.
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$Python3 = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python313\python.exe'
if (-not (Test-Path -LiteralPath $Python3)) {
    throw 'Install Python 3.13 or edit $Python3 to an existing Python 3.10-3.13 executable.'
}
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & $Python3 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}
$ProjectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
# cu118 works with this laptop driver 529.04; do not install a CUDA 13 wheel.
# Pin paired versions per https://pytorch.org/get-started/previous-versions/
& $ProjectPython -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
& $ProjectPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Project dependency installation failed.' }
& $ProjectPython -m ipykernel install --prefix (Join-Path $PSScriptRoot '.venv') --name bird-local-gpu --display-name 'Python 3 (.venv GPU)'
if ($LASTEXITCODE -ne 0) { throw 'Project-local Jupyter kernel installation failed.' }
& $ProjectPython -c "import torch; print(torch.__version__, torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA is not available'; print(torch.cuda.get_device_name(0)); print((torch.ones(8,device='cuda')**2).sum().item())"
if ($LASTEXITCODE -ne 0) { throw 'GPU verification failed. Check the NVIDIA driver and Python environment.' }
Write-Host 'Ready. In VS Code select .venv\Scripts\python.exe as the notebook kernel.'
Write-Host 'Or run: .venv\Scripts\python.exe -m jupyterlab'
