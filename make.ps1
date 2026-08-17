<#
.SYNOPSIS
    Windows shim for the Makefile targets, so the documented workflow is the
    same on the dev laptop and on the Linux GPU server.

.EXAMPLE
    .\make.ps1 test
    .\make.ps1 setup -Torch cu124
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('help', 'setup', 'quality', 'fmt', 'test', 'test-all', 'smoke',
                 'train-dev', 'train-full', 'bench', 'arena', 'calibrate',
                 'plots', 'demo', 'clean')]
    [string]$Target = 'help',

    [string]$Torch = 'cpu',
    [string]$Config = 'configs/dev8x8.yaml'
)

$ErrorActionPreference = 'Stop'
$uv = 'uv'

function Invoke-Step {
    param([string[]]$Command)
    Write-Host "==> $($Command -join ' ')" -ForegroundColor Cyan
    & $Command[0] @($Command[1..($Command.Length - 1)])
    if ($LASTEXITCODE -ne 0) { throw "failed: $($Command -join ' ')" }
}

switch ($Target) {
    'help' {
        Write-Host 'Targets: setup quality fmt test test-all smoke train-dev train-full bench arena calibrate plots demo clean'
        Write-Host 'Example: .\make.ps1 setup -Torch cu124'
    }
    'setup' {
        Invoke-Step @($uv, 'sync', '--extra', $Torch, '--extra', 'dev', '--extra', 'api', '--extra', 'obs')
        if (Test-Path 'web/package.json') { Invoke-Step @('npm', '--prefix', 'web', 'ci') }
    }
    'quality' {
        Invoke-Step @($uv, 'run', 'ruff', 'check', '.')
        Invoke-Step @($uv, 'run', 'ruff', 'format', '--check', '.')
        Invoke-Step @($uv, 'run', 'pyright')
    }
    'fmt' {
        Invoke-Step @($uv, 'run', 'ruff', 'check', '--fix', '.')
        Invoke-Step @($uv, 'run', 'ruff', 'format', '.')
    }
    'test'      { Invoke-Step @($uv, 'run', 'pytest', '-m', 'not slow and not gpu') }
    'test-all'  { Invoke-Step @($uv, 'run', 'pytest', '-m', 'not gpu') }
    'smoke'     { Invoke-Step @($uv, 'run', 'reversi', 'train', '-c', 'configs/smoke4x4.yaml') }
    'train-dev' { Invoke-Step @($uv, 'run', 'reversi', 'train', '-c', 'configs/dev8x8.yaml', '--resume', 'auto') }
    'train-full'{ Invoke-Step @($uv, 'run', 'reversi', 'train', '-c', 'configs/full8x8.yaml', '--resume', 'auto') }
    'bench'     { Invoke-Step @($uv, 'run', 'reversi', 'bench', '-c', $Config) }
    'arena'     { Invoke-Step @($uv, 'run', 'reversi', 'arena', '-c', $Config) }
    'calibrate' { Invoke-Step @($uv, 'run', 'reversi', 'calibrate', '-c', $Config, '--validate') }
    'plots'     { Invoke-Step @($uv, 'run', 'python', 'scripts/make_plots.py') }
    'demo' {
        Invoke-Step @($uv, 'run', 'python', 'scripts/download_model.py')
        Invoke-Step @('npm', '--prefix', 'web', 'run', 'build')
        Invoke-Step @($uv, 'run', 'reversi', 'serve')
    }
    'clean' {
        foreach ($p in @('.pytest_cache', '.ruff_cache', '.coverage', 'coverage.xml', 'htmlcov')) {
            if (Test-Path $p) { Remove-Item -Recurse -Force $p }
        }
        Get-ChildItem -Recurse -Directory -Filter '__pycache__' |
            ForEach-Object { Remove-Item -Recurse -Force $_.FullName }
    }
}
