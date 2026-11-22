import argparse
import pickle
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with args.snapshot.open("rb") as stream:
        snapshot = pickle.load(stream)

    events = snapshot["device_traces"][0]
    active = {}
    current = 0
    peak = 0
    peak_active = {}
    action_counts = Counter()

    for event in events:
        action = event.get("action")
        action_counts[action] += 1
        addr = event.get("addr")
        size = int(event.get("size") or 0)
        if action == "alloc":
            active[addr] = event
            current += size
            if current > peak:
                peak = current
                peak_active = dict(active)
        elif action in ("free", "free_requested", "free_completed"):
            previous = active.pop(addr, None)
            if previous is not None:
                current -= int(previous.get("size") or 0)

    reserved = sum(int(segment.get("total_size") or 0) for segment in snapshot["segments"])
    output = args.output or args.snapshot.with_name("kohya-memory-analysis.txt")

    size_counts = Counter(int(event.get("size") or 0) for event in peak_active.values())
    frame_counts = defaultdict(lambda: {"count": 0, "bytes": 0})
    for event in peak_active.values():
        frames = event.get("frames") or []
        frame = frames[0] if frames else {}
        key = f"{frame.get('filename', '<unknown>')}:{frame.get('line', '?')} in {frame.get('name', '?')}"
        frame_counts[key]["count"] += 1
        frame_counts[key]["bytes"] += int(event.get("size") or 0)

    lines = [
        f"Snapshot: {args.snapshot}",
        f"Events: {len(events)}",
        f"Actions: {dict(action_counts)}",
        f"Peak allocated: {peak / 1024**3:.3f} GiB",
        f"Reserved segments: {reserved / 1024**3:.3f} GiB",
        f"Active blocks at peak: {len(peak_active)}",
        "",
        "Top buffer sizes at peak:",
    ]
    for size, count in sorted(size_counts.items(), key=lambda item: item[0] * item[1], reverse=True)[:20]:
        lines.append(f"  {size / 1024**2:10.3f} MiB x {count:5d} = {size * count / 1024**2:10.3f} MiB")
    lines += ["", "Top allocation call sites at peak:"]
    for key, stats in sorted(frame_counts.items(), key=lambda item: item[1]["bytes"], reverse=True)[:20]:
        lines.append(f"  {stats['bytes'] / 1024**2:10.3f} MiB / {stats['count']:5d} blocks  {key}")

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
