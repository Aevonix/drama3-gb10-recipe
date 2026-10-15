"""TTFB, RTF and steady-state decode throughput for a Drama 3 service.

Decode throughput is the number the byte model predicts on, so it is the one to
watch across quantization changes. It is only available from the engine log, so
this needs read access to that log as well as to the endpoint.

The first decode window of every burst is dropped: it covers a partial window
right after prefill and reads around 3.9 tok/s, which drags any average.

  DRAMA3_URL=http://host:18020/v1/audio/speech \
  DRAMA3_LOG=/path/to/engine.log \
  DRAMA3_REF=/path/on/the/serving/node/reference.wav \
  DRAMA3_REF_TEXT='the transcript of that reference' \
    python3 scripts/bench.py "label for this run"

DRAMA3_REF is resolved on the serving node, not here. Use your own recording or
a repository example; see NOTICE on whose voice a reference may be.
"""
import json, os, re, statistics as st, subprocess, sys, time, urllib.request

URL = os.environ.get("DRAMA3_URL", "http://127.0.0.1:18020/v1/audio/speech")
LOG = os.environ.get("DRAMA3_LOG", "")
REF = os.environ.get("DRAMA3_REF", "")
REF_TEXT = os.environ.get("DRAMA3_REF_TEXT", "")
OUT = os.environ.get("DRAMA3_BENCH_JSON", "decode-bench.json")

SHORT = "The index is rebuilt and the service is answering again."
LONG = ("The first run took nine minutes, which told us nothing except that the "
        "cache was cold. The second run took forty seconds, and that is the "
        "number worth writing down.")

RATE = re.compile(r"gen throughput \(token/s\): ([0-9.]+)")
TOKENS = re.compile(r"#full token: (\d+)")


def log_lines():
    if not LOG:
        return 0
    return int(subprocess.run(["wc", "-l", LOG], capture_output=True, text=True,
                              check=True).stdout.split()[0])


def decode_rates(start):
    """Steady-state decode rates since line `start`, first window of each burst dropped."""
    if not LOG:
        return [], []
    text = subprocess.run(f"tail -n +{start + 1} {LOG!r} | grep 'Decode batch'",
                          shell=True, capture_output=True, text=True).stdout
    bursts, current, previous = [], [], None
    for line in text.splitlines():
        rate = RATE.search(line)
        if not rate:
            continue
        seen = TOKENS.search(line)
        seen = int(seen.group(1)) if seen else None
        # A burst restarts when the token count drops: that is a new request.
        if previous is not None and seen is not None and seen < previous:
            if current:
                bursts.append(current)
            current = []
        current.append(float(rate.group(1)))
        previous = seen
    if current:
        bursts.append(current)
    return bursts, [v for b in bursts for v in b[1:]]


def render(text):
    """One request; returns time to first audio bytes and wall-clock / audio seconds."""
    body = {"input": text, "response_format": "wav", "max_new_tokens": 256,
            "temperature": 0.7, "top_p": 0.9, "top_k": 50}
    if REF:
        body["references"] = [{"audio_path": REF, "text": REF_TEXT}]
    request = urllib.request.Request(
        URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    start = time.monotonic()
    ttfb, size = None, 0
    with urllib.request.urlopen(request) as response:
        while True:
            chunk = response.read(2048)
            if not chunk:
                break
            if ttfb is None:
                ttfb = time.monotonic() - start
            size += len(chunk)
    wall = time.monotonic() - start
    seconds = max(size - 44, 0) / (44100 * 2)  # 44.1 kHz mono 16-bit
    return ttfb, (wall / seconds if seconds else None)


def passes(text, n=4):
    results = [render(text) for _ in range(n)]
    return (round(st.median(t for t, _ in results), 3),
            round(st.median(r for _, r in results if r), 3))


def main(label):
    at = log_lines()
    short_ttfb, short_rtf = passes(SHORT)
    long_ttfb, long_rtf = passes(LONG)
    bursts, steady = decode_rates(at)
    row = {"label": label, "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
           "short_ttfb_s": short_ttfb, "short_rtf": short_rtf,
           "long_ttfb_s": long_ttfb, "long_rtf": long_rtf,
           "decode_tok_s_median": round(st.median(steady), 2) if steady else None,
           "decode_tok_s_n": len(steady),
           "decode_tok_s_spread": [round(min(steady), 2), round(max(steady), 2)] if steady else None,
           "bursts": len(bursts)}
    try:
        rows = json.load(open(OUT))
    except Exception:
        rows = []
    rows.append(row)
    json.dump(rows, open(OUT, "w"), indent=1)
    print(json.dumps(row, indent=1))
    if not steady:
        print("\nNo decode windows found. Set DRAMA3_LOG to the engine log to get "
              "the throughput number; TTFB and RTF above are still valid.", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "unlabeled")
