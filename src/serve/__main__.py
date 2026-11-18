"""Run the model server.

Usage: uv run python -m serve [--corpus data/corpus] [--results results]
                              [--port 8100] [--streaming 8] [--batch 4]

Listens on 127.0.0.1 only: the website talks to it, browsers do not.
"""

import argparse
from pathlib import Path

from serve.app import create_app

ROOT: Path = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m serve")
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "corpus")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--port", type=int, default=8100)
    # OWN CHOICE: 8 streaming + 4 batch processes of the 16 threads.
    parser.add_argument("--streaming", type=int, default=8)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()
    app = create_app(args.corpus, args.results, ROOT / "src",
                     args.streaming, args.batch)
    # threaded: long polls and many sessions each hold a thread. No
    # reloader and one process: sessions live in this process's memory.
    app.run(host="127.0.0.1", port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
