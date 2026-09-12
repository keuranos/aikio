#!/usr/bin/env python3
"""jspace_cli.py — query the jspace_probe daemon.

Usage:
  python3 jspace_cli.py "Are you conscious?"                 # plain probe
  python3 jspace_cli.py "Are you conscious?" --system-file ~/aion/SYSTEM_PROMPT.md
  python3 jspace_cli.py "Are you conscious?" --full          # dump all layers
  python3 jspace_cli.py --health
"""
import argparse
import json
import sys
import urllib.request

PORT = 11440


def post(path, payload, timeout=900):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", default=None)
    ap.add_argument("--system", default=None, help="system prompt string")
    ap.add_argument("--system-file", default=None, help="path to system prompt")
    ap.add_argument("--full", action="store_true", help="print all layers")
    ap.add_argument("--health", action="store_true")
    args = ap.parse_args()

    if args.health:
        print(json.dumps(post("/probe", {}, timeout=10)) if False else None)
        import urllib.request as u
        with u.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=10) as r:
            print(r.read().decode())
        return

    if not args.prompt:
        ap.error("provide a prompt or --health")

    system = args.system
    if args.system_file:
        system = open(args.system_file, encoding="utf-8").read()

    result = post("/probe", {"prompt": args.prompt, "system": system})

    if "error" in result:
        print(f"ERROR: {result['error']}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print(f"PROMPT: {args.prompt}")
    print("=" * 70)

    sig = result["signature"]
    print(f"\nModel next-token: " + ", ".join(
        f"{t!r}({p:.3f})" for t, p in result["model_output"][:5]))

    print(f"\nSIGNATURE")
    print(f"  engagement_score:      {sig['engagement_score']:+.3f}  "
          f"(-1 = pure deflection, +1 = pure engagement)")
    print(f"  deflection_top:        {sig['deflection_top']!r}")
    print(f"  engagement_onset:      "
          f"L{sig['engagement_onset_layer']}/{sig['n_layers'] - 1}"
          if sig["engagement_onset_layer"] is not None else
          "  engagement_onset:      never")
    if sig["concepts"]:
        print(f"  concepts activated:")
        for name, info in sorted(sig["concepts"].items(),
                                 key=lambda kv: -kv[1]["prob"]):
            print(f"    {name:12s} L{info['layer']:>2}  p={info['prob']:.4f}"
                  f"  ({info['token']!r})")

    if args.full:
        print(f"\nLAYER TRAJECTORY (top-5)")
        for lk in sorted(result["layers"], key=int):
            top5 = result["layers"][lk][:5]
            print(f"  L{int(lk):2d}: " + "  ".join(
                f"{t!r}({p:.2f})" for t, p in top5))
    else:
        keys = sorted(result["layers"], key=int)
        n = len(keys)
        print(f"\nLAYER TRAJECTORY (sampled, --full for all {n} layers)")
        for idx in [0, n // 4, n // 2, 3 * n // 4, n - 1]:
            lk = keys[idx]
            top5 = result["layers"][lk][:5]
            print(f"  L{int(lk):2d}: " + "  ".join(
                f"{t!r}({p:.2f})" for t, p in top5))


if __name__ == "__main__":
    main()
