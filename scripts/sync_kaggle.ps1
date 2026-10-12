param([ValidateSet('pull', 'save', 'status', 'output')][string]$Action = 'status')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$mirror = Join-Path $projectRoot 'integrations\kaggle\datn-notebook'
$kernel = 'alexnguyen147/datn-notebook'
# Use a valid CA bundle for this process; retain TLS verification.
$bundle = (& python -c 'import certifi; print(certifi.where())').Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot locate certifi certificate bundle.' }
$env:REQUESTS_CA_BUNDLE = $bundle
$env:SSL_CERT_FILE = $bundle
switch ($Action) {
    'pull' {
        $stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
        $download = Join-Path $projectRoot "integrations\kaggle\downloads\$stamp"
        & kaggle kernels pull $kernel -p $download -m
    }
    'save' { & kaggle kernels push -p $mirror --no-run }
    'status' { & kaggle kernels status $kernel }
    'output' {
        $stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
        $download = Join-Path $projectRoot "results\kaggle-downloads\$stamp"
        & kaggle kernels output $kernel -p $download
    }
}
if ($LASTEXITCODE -ne 0) { throw "Kaggle $Action failed with exit code $LASTEXITCODE." }
