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
$structured = [System.Collections.Generic.List[object]]::new()
$phaseMemory = @{}

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

    if ($line -match '\[cubecl-memory\].*event=(\w+) pool=(\w+) requested=(\d+) in_use=(\d+) reserved=(\d+) allocations=(\d+) released_in_use=(\d+) released_reserved=(\d+)') {
        $row = [pscustomobject]@{
            Phase = $phase
            Event = $Matches[1]
            Pool = $Matches[2]
            Requested = [int64]$Matches[3]
            InUse = [int64]$Matches[4]
            Reserved = [int64]$Matches[5]
            Allocations = [int64]$Matches[6]
            ReleasedInUse = [int64]$Matches[7]
            ReleasedReserved = [int64]$Matches[8]
        }
        $structured.Add($row)
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
$result.Add("")
$result.Add("## Structured allocation events by phase, event, and pool")
$grouped = $structured | Group-Object Phase, Event, Pool
foreach ($group in ($grouped | Sort-Object Name)) {
    $rows = @($group.Group)
    $result.Add(("{0}`tcount={1}`trequested_sum={2}`treleased_in_use={3}`treleased_reserved={4}`tmax_in_use={5}`tlast_in_use={6}`tmax_reserved={7}" -f `
        $group.Name,
        $rows.Count,
        (($rows | Measure-Object Requested -Sum).Sum),
        (($rows | Measure-Object ReleasedInUse -Sum).Sum),
        (($rows | Measure-Object ReleasedReserved -Sum).Sum),
        (($rows | Measure-Object InUse -Maximum).Maximum),
        $rows[-1].InUse,
        (($rows | Measure-Object Reserved -Maximum).Maximum)))
}
$result.Add("")
$result.Add("## New allocation sizes")
$newGroups = $structured | Where-Object Event -eq "allocate_new" | Group-Object Pool, Requested
foreach ($group in ($newGroups | Sort-Object {[int64](($_.Name -split ', ')[1])}, Name)) {
    $result.Add(("{0}`tcount={1}" -f $group.Name, $group.Count))
}
$result.Add("")
$result.Add("## Cleanup releases")
foreach ($row in ($structured | Where-Object Event -like "cleanup*")) {
    $result.Add(("{0}`tpool={1}`treleased_in_use={2}`treleased_reserved={3}`tin_use={4}`treserved={5}" -f `
        $row.Phase, $row.Pool, $row.ReleasedInUse, $row.ReleasedReserved, $row.InUse, $row.Reserved))
}

if ($OutputPath) {
    Set-Content -LiteralPath $OutputPath -Value $result -Encoding utf8
} else {
    $result
}
