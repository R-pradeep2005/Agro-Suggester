Write-Host "=========================================="
Write-Host "Starting Agro-Suggester Services Locally"
Write-Host "Make sure to close the spawned windows to stop backend services"
Write-Host "==========================================`n"

Write-Host "[0/5] Setting up Virtual Environment..."
if (!(Test-Path -Path "venv")) {
  python -m venv venv
}

$pythonPath = "$PSScriptRoot\venv\Scripts\python.exe"
$pipPath = "$PSScriptRoot\venv\Scripts\pip.exe"

# 1. Recommendation Service
Write-Host "[1/5] Setting up Recommendation Service..."
Start-Process -FilePath "powershell" -ArgumentList "-NoExit", "-Command", "`$Host.UI.RawUI.WindowTitle = 'Recommendation Service (8002)'; cd recommendation; & '$pipPath' install -r requirements.txt; & '$pythonPath' -m uvicorn app.main:app --host 0.0.0.0 --port 8002"
Write-Host "      Recommendation Service running on port 8002"

# 2. Input Prep Service
Write-Host "[2/5] Setting up Input Prep Service..."
Start-Process -FilePath "powershell" -ArgumentList "-NoExit", "-Command", "`$Host.UI.RawUI.WindowTitle = 'Input Prep Service (8001)'; cd input_prep; & '$pipPath' install -r requirements.txt; & '$pythonPath' -m uvicorn app.main:app --host 0.0.0.0 --port 8001"
Write-Host "      Input Prep Service running on port 8001"

# 3. API Gateway
Write-Host "[3/5] Setting up API Gateway..."
Start-Process -FilePath "powershell" -ArgumentList "-NoExit", "-Command", "`$Host.UI.RawUI.WindowTitle = 'API Gateway (8000)'; cd gateway; & '$pipPath' install -r requirements.txt; & '$pythonPath' -m uvicorn app.main:app --host 0.0.0.0 --port 8000"
Write-Host "      API Gateway running on port 8000"

# 4. React Frontend
Write-Host "[4/5] Setting up Frontend..."
cd frontend
npm install --silent
Write-Host "      Frontend starting on http://localhost:5173"
Write-Host "=========================================="

$env:VITE_API_GATEWAY="http://localhost:8000"
npm run dev
