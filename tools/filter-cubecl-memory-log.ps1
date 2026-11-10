param(
    [Parameter(Mandatory = $true)]
    [string] $InputPath,
    [string] $OutputPath = ""
)

$phase = "startup"
$events = [System.Collections.Generic.List[string]]::new()
$phaseCounts = @{}
$sizeCounts = @{}
$freeCounts = @{}

foreach ($line in Get-Content -LiteralPath $InputPath) {
    if ($line -match '\[memory-phase\] step=(\d+) phase=([^ ]+)') {
        $phase = "step=$($Matches[1])/$($Matches[2])"
        $events.Add("PHASE $phase")
        continue
    }

    if ($line -match 'Reserved memory ([0-9.]+ (?:B|KB|MB|GB)) using dynamic pool') {
        $size = $Matches[1]
        $key = "$phase|$size"
        $sizeCounts[$key] = 1 + ($sizeCounts[$key] ?? 0)
        continue
    }

    if ($line -match 'Allocated a new memory page') {
        $phaseCounts[$phase] = 1 + ($phaseCounts[$phase] ?? 0)
        $events.Add("PAGE phase=$phase")
        continue
    }

    if ($line -match 'Page .+ is_free=(true|false)') {
        $key = "$phase|$($Matches[1])"
        $freeCounts[$key] = 1 + ($freeCounts[$key] ?? 0)
        continue
    }

    if ($line -match '\[memory\] step=') {
        $events.Add($line.Trim())
        continue
    }
}

$result = [System.Collections.Generic.List[string]]::new()
$result.Add("# CubeCL memory trace: $InputPath")
$result.Add("## Phase markers and step summaries")
$result.AddRange($events)
$result.Add("")
$result.Add("## New page allocations by phase")
foreach ($entry in ($phaseCounts.GetEnumerator() | Sort-Object Name)) {
    $result.Add("$($entry.Name)`t$($entry.Value)")
}
$result.Add("")
$result.Add("## Dynamic reservations by phase and size")
foreach ($entry in ($sizeCounts.GetEnumerator() | Sort-Object Name)) {
    $result.Add("$($entry.Name)`t$($entry.Value)")
}
$result.Add("")
$result.Add("## Page state observations by phase")
foreach ($entry in ($freeCounts.GetEnumerator() | Sort-Object Name)) {
    $result.Add("$($entry.Name)`t$($entry.Value)")
}

if ($OutputPath) {
    Set-Content -LiteralPath $OutputPath -Value $result -Encoding utf8
} else {
    $result
}
