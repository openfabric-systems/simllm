[CmdletBinding()]
param(
    [string]$BuildDirectory = "build\htsim",
    [ValidateSet("Debug", "Release", "RelWithDebInfo", "MinSizeRel")]
    [string]$Configuration = "Release",
    [ValidateRange(1, 256)]
    [int]$Jobs = 4,
    [switch]$RunTests
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$sourceDirectory = Join-Path $repoRoot "third_party\htsim\htsim\sim"
if (-not [IO.Path]::IsPathRooted($BuildDirectory)) {
    $BuildDirectory = Join-Path $repoRoot $BuildDirectory
}
$enableTests = if ($RunTests) { "ON" } else { "OFF" }

& cmake -S $sourceDirectory -B $BuildDirectory `
    "-DENABLE_TESTS=$enableTests" `
    "-DHTSIM_CREATE_SOURCE_SYMLINKS=OFF"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& cmake --build $BuildDirectory --config $Configuration --parallel $Jobs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($RunTests) {
    & ctest --test-dir $BuildDirectory -C $Configuration `
        --output-on-failure --parallel $Jobs
    exit $LASTEXITCODE
}
