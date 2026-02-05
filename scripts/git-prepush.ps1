param(
  [string]$Remote = "origin",
  [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"

function RunGit([string[]]$Args) {
  & git @Args
  return $LASTEXITCODE
}

function GetUnmergedFiles() {
  $out = & git diff --name-only --diff-filter=U
  if ($LASTEXITCODE -ne 0) { return @() }
  return ($out | Where-Object { $_ -and $_.Trim() -ne "" })
}

function TryResolveWorkflowCronConflict([string]$Path) {
  $ours = & git show (":2:$Path") 2>$null
  if ($LASTEXITCODE -ne 0) { return $false }
  $theirs = & git show (":3:$Path") 2>$null
  if ($LASTEXITCODE -ne 0) { return $false }

  $cronRe = '(?m)^\s*-\s*cron:\s*"[^"]+"\s*#\s*self-adapt\s+level=\d+\s*$'
  $theirCron = [regex]::Match($theirs, $cronRe).Value
  if (-not $theirCron) { return $false }

  if (-not [regex]::IsMatch($ours, $cronRe)) { return $false }

  $resolved = [regex]::Replace($ours, $cronRe, $theirCron, 1)
  Set-Content -LiteralPath $Path -Value $resolved -Encoding UTF8 -NoNewline

  & git add -- $Path | Out-Null
  return ($LASTEXITCODE -eq 0)
}

function IsDirtyWorkTree() {
  $out = & git status --porcelain
  if ($LASTEXITCODE -ne 0) { return $true }
  return (($out | Measure-Object).Count -gt 0)
}

if (-not (& git rev-parse --is-inside-work-tree 2>$null)) {
  throw "Not inside a git work tree."
}

if (IsDirtyWorkTree) {
  Write-Host "Work tree has local changes. Please commit/stash first." -ForegroundColor Yellow
  Write-Host "Hint: git status" -ForegroundColor Yellow
  exit 2
}

Write-Host "Fetching $Remote ..."
$rc = RunGit @("fetch", $Remote)
if ($rc -ne 0) { exit $rc }

Write-Host "Rebasing onto $Remote/$Branch ..."
$rc = RunGit @("rebase", "$Remote/$Branch")

while ($rc -ne 0) {
  $unmerged = GetUnmergedFiles
  if (-not $unmerged -or $unmerged.Count -eq 0) {
    Write-Host "Rebase failed, but no unmerged files found. Resolve manually." -ForegroundColor Red
    exit 3
  }

  $remaining = @()
  foreach ($f in $unmerged) {
    $isWorkflow = ($f -eq ".github/workflows/hf-requests-ingest.yml") -or ($f -eq ".github/workflows/hf-auto-merge-additive-prs.yml")
    if (-not $isWorkflow) {
      $remaining += $f
      continue
    }

    Write-Host "Auto-resolving workflow cron conflict: $f"
    $ok = $false
    try { $ok = TryResolveWorkflowCronConflict $f } catch { $ok = $false }

    if (-not $ok) {
      $remaining += $f
    }
  }

  if ($remaining.Count -gt 0) {
    Write-Host "Unresolved conflicts remain:" -ForegroundColor Red
    $remaining | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    Write-Host "Resolve them, then run: git rebase --continue" -ForegroundColor Yellow
    exit 4
  }

  Write-Host "Continuing rebase ..."
  $rc = RunGit @("rebase", "--continue")
}

Write-Host "Rebase complete. Pushing ..."
$rc = RunGit @("push", $Remote, "HEAD:$Branch")
exit $rc
