param(
    [ValidateSet("precheck", "smoke", "custom")]
    [string]$Mode = "precheck",

    [int]$Steps = 0,
    [int]$BatchSize = 2,
    [string]$Device = "cuda",
    [double]$SnrMin = 20.0,
    [double]$SnrMax = 30.0,
    [double]$MaxDelaySamples = 3.0,
    [double]$MaxDopplerHz = 500.0,
    [int]$EvalEvery = 0,
    [int]$EvalBatches = 1,
    [int]$SaveEvery = 0,
    [string]$OutputTag = "",
    [string]$PythonExe = "E:\pytorch\python.exe",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")

if ($Mode -eq "precheck") {
    if ($Steps -le 0) { $Steps = 10 }
    if ($EvalEvery -le 0) { $EvalEvery = 5 }
    if ($SaveEvery -le 0) { $SaveEvery = 5 }
    if ([string]::IsNullOrWhiteSpace($OutputTag)) { $OutputTag = "precheck_${Steps}step" }
}
elseif ($Mode -eq "smoke") {
    if ($Steps -le 0) { $Steps = 50 }
    if ($EvalEvery -le 0) { $EvalEvery = 10 }
    if ($SaveEvery -le 0) { $SaveEvery = 25 }
    if ([string]::IsNullOrWhiteSpace($OutputTag)) { $OutputTag = "smoke_${Steps}step" }
}
else {
    if ($Steps -le 0) { $Steps = 20 }
    if ($EvalEvery -le 0) { $EvalEvery = [Math]::Max(1, [Math]::Floor($Steps / 5)) }
    if ($SaveEvery -le 0) { $SaveEvery = [Math]::Max(1, [Math]::Floor($Steps / 2)) }
    if ([string]::IsNullOrWhiteSpace($OutputTag)) { $OutputTag = "custom_${Steps}step" }
}

if ($EvalBatches -le 0) {
    throw "EvalBatches must be positive."
}

$OutputDir = Join-Path $ProjectRoot "outputs\stage7_perceiver_receiver_smoke_train\$OutputTag"
$TrainScript = Join-Path $ProjectRoot "train_dd_token_perceiver_receiver.py"

$Arguments = @(
    $TrainScript,
    "--output-dir", $OutputDir,
    "--codebook-size", "256",
    "--token-shape", "16", "16",
    "--symbols-per-token", "4",
    "--dd-shape", "32", "32",
    "--cp-len", "4",
    "--batch-size", "$BatchSize",
    "--num-steps", "$Steps",
    "--embed-dim", "128",
    "--num-heads", "4",
    "--self-attn-layers", "2",
    "--snr-db-min", "$SnrMin",
    "--snr-db-max", "$SnrMax",
    "--num-paths", "3",
    "--max-delay-samples", "$MaxDelaySamples",
    "--max-doppler-hz", "$MaxDopplerHz",
    "--device", $Device,
    "--eval-every", "$EvalEvery",
    "--eval-batches", "$EvalBatches",
    "--save-every", "$SaveEvery"
)

if ($DryRun) {
    $Arguments += "--dry-run"
}

Write-Host "Project root: $ProjectRoot"
Write-Host "Output dir:   $OutputDir"
Write-Host "Mode:         $Mode"
Write-Host "Steps:        $Steps"
Write-Host "Device:       $Device"
Write-Host "Eval/Save:    every $EvalEvery / $SaveEvery steps, eval batches $EvalBatches"
Write-Host "SNR:          $SnrMin to $SnrMax dB"
Write-Host "Delay/Doppler $MaxDelaySamples samples / $MaxDopplerHz Hz"
Write-Host ""
Write-Host "Running: $PythonExe $($Arguments -join ' ')"

Push-Location $ProjectRoot
try {
    & $PythonExe @Arguments
}
finally {
    Pop-Location
}
