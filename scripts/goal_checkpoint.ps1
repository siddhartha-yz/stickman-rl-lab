[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$Goal,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string[]]$Paths,

    [string]$Python,
    [string]$Remote = "origin",
    [switch]$NoPush,
    [switch]$SkipChecks
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-Git {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    $output = & git @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "git $($Arguments -join ' ') failed:`n$($output -join [Environment]::NewLine)"
    }
    return @($output)
}

function Resolve-Python {
    param([string]$RequestedPython, [string]$RepositoryRoot)

    $candidates = @()
    if ($RequestedPython) {
        $candidates += $RequestedPython
    }
    $candidates += (Join-Path $RepositoryRoot ".venv\Scripts\python.exe")
    $candidates += "C:\rlv\Scripts\python.exe"

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    $command = Get-Command python.exe -ErrorAction SilentlyContinue
    if (-not $command) {
        $command = Get-Command python -ErrorAction SilentlyContinue
    }
    if (-not $command) {
        throw "No Python interpreter was found. Pass -Python with an absolute interpreter path."
    }
    return $command.Source
}

$repositoryRoot = (Invoke-Git -Arguments @("rev-parse", "--show-toplevel") | Select-Object -First 1).Trim()
if (-not $repositoryRoot) {
    throw "The current directory is not inside a Git repository."
}
$repositoryRoot = [IO.Path]::GetFullPath($repositoryRoot)
Set-Location -LiteralPath $repositoryRoot

$branch = (Invoke-Git -Arguments @("branch", "--show-current") | Select-Object -First 1).Trim()
if (-not $branch) {
    throw "Detached HEAD is not supported. Check out a named branch first."
}

# `powershell.exe -File` can bind comma-separated CLI values as one array item.
$Paths = @(
    $Paths |
        ForEach-Object { $_ -split ',' } |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ }
)
if ($Paths.Count -eq 0) {
    throw "At least one checkpoint path is required."
}

& git diff --cached --quiet
$indexStatus = $LASTEXITCODE
if ($indexStatus -eq 1) {
    throw "The Git index already contains staged changes. Commit or unstage them before creating an automatic goal checkpoint."
}
if ($indexStatus -ne 0) {
    throw "Unable to inspect the Git index."
}

$rootPrefix = $repositoryRoot.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
$safePaths = @()
foreach ($path in $Paths) {
    if ([string]::IsNullOrWhiteSpace($path)) {
        throw "Checkpoint paths cannot be empty."
    }

    $fullPath = [IO.Path]::GetFullPath((Join-Path $repositoryRoot $path))
    if (-not $fullPath.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path escapes the repository: $path"
    }

    $normalized = $path.Replace('/', '\')
    if ($normalized -match '(?i)(^|\\)(\.env($|\.)|[^\\]*\.(pem|key|p12|pfx)$|credentials[^\\]*\.json$|service-account[^\\]*\.json$)') {
        throw "Refusing to stage a likely secret: $path"
    }

    & git check-ignore -q -- $path
    if ($LASTEXITCODE -eq 0) {
        throw "Refusing to stage an ignored path: $path"
    }
    if ($LASTEXITCODE -gt 1) {
        throw "Unable to check ignore rules for: $path"
    }

    if (-not (Test-Path -LiteralPath $fullPath)) {
        & git ls-files --error-unmatch -- $path *> $null
        if ($LASTEXITCODE -ne 0) {
            throw "Path does not exist and is not a tracked deletion: $path"
        }
    }
    $safePaths += $path
}

if (-not $SkipChecks) {
    $pythonExecutable = Resolve-Python -RequestedPython $Python -RepositoryRoot $repositoryRoot

    & $pythonExecutable -m ruff check src scripts tests
    if ($LASTEXITCODE -ne 0) {
        throw "Ruff failed; no files were staged."
    }

    & $pythonExecutable -m pytest -q
    if ($LASTEXITCODE -ne 0) {
        throw "Pytest failed; no files were staged."
    }
}

Invoke-Git -Arguments (@("add", "--") + $safePaths) | Out-Null

& git diff --cached --quiet
if ($LASTEXITCODE -eq 0) {
    throw "The declared goal paths contain no staged changes."
}
if ($LASTEXITCODE -ne 1) {
    throw "Unable to inspect the staged checkpoint."
}

$stagedPaths = Invoke-Git -Arguments @("diff", "--cached", "--name-only", "--diff-filter=ACDMRTUXB")
$cleanGoal = ($Goal -replace '[\r\n]+', ' ').Trim()
$message = "feat(goal): $cleanGoal"

Invoke-Git -Arguments @("commit", "-m", $message) | Out-Null
$commitHash = (Invoke-Git -Arguments @("rev-parse", "HEAD") | Select-Object -First 1).Trim()

if (-not $NoPush) {
    Invoke-Git -Arguments @("push", "-u", $Remote, "HEAD") | Out-Null
}

Write-Host "Goal checkpoint created successfully."
Write-Host "Branch: $branch"
Write-Host "Commit: $commitHash"
Write-Host "Message: $message"
Write-Host "Files:"
$stagedPaths | ForEach-Object { Write-Host "  $_" }
Write-Host "Pushed: $(-not $NoPush)"
