"""Worker processes.

Streaming decoders keep state between fixes, so each (session, model)
lives in one fixed worker process and every fix for it goes there in
order: N processes, one Pipe each (a process pool cannot pin a task to a
process). Batch matches for `at_finish` models are stateless and go to a
separate process pool, so a slow one cannot hold up anyone's fixes.
"""

import multiprocessing as mp
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from serve import models


def serve_forever(connection, results: str) -> None:
    """A streaming worker: (command, key, *arguments) in, (ok, value) out."""
    decoders: dict[str, object] = {}
    while True:
        try:
            command, key, *arguments = connection.recv()
        except (EOFError, OSError):  # the server process has gone
            return
        try:
            if command == "open":
                model, map_dir = arguments
                decoders[key] = (models.open_online(model, map_dir, results),
                                 map_dir)
                value = None
            elif command == "push":
                (fixes,) = arguments
                decoder, map_dir = decoders[key]
                update = decoder.push(fixes)
                g = models.graph(map_dir)
                value = {"kept": update.kept,
                         "committed": update.committed,
                         "tentative": update.tentative,
                         "pending_fixes": update.pending_fixes,
                         "break": update.new_trip,
                         "geometry": {
                             "committed": models.geometry(g, update.committed),
                             "tentative": models.geometry(g, update.tentative)}}
            elif command == "finish":
                (truth,) = arguments
                decoder, map_dir = decoders.pop(key)
                route, counts = decoder.finish()
                value = models.result(models.graph(map_dir), route, counts, truth)
            elif command == "close":
                decoders.pop(key, None)
                value = None
            else:
                raise ValueError(f"unknown command {command!r}")
            connection.send((True, value))
        except Exception as error:  # reported to the session, worker lives on
            connection.send((False, f"{type(error).__name__}: {error}"))


class Worker:
    def __init__(self, context, results: Path) -> None:
        self.connection, child = context.Pipe()
        self.process = context.Process(target=serve_forever,
                                       args=(child, str(results)), daemon=True)
        self.process.start()
        self.lock = threading.Lock()
        self.keys: set[str] = set()
        self.maps: set[str] = set()

    def call(self, command: str, key: str, *arguments):
        with self.lock:
            self.connection.send((command, key, *arguments))
            ok, value = self.connection.recv()
        if not ok:
            raise RuntimeError(value)
        return value


class Workers:
    def __init__(self, results: Path, streaming: int, batch: int) -> None:
        context = mp.get_context("spawn")  # never fork a process holding CUDA
        self.streaming: list[Worker] = [Worker(context, results)
                                        for _ in range(streaming)]
        self.batch = ProcessPoolExecutor(batch, mp_context=context)
        self.placed: dict[str, Worker] = {}
        self.lock = threading.Lock()

    def place(self, key: str, map_id: str) -> Worker:
        """A worker that already has the map parsed, else the least busy."""
        with self.lock:
            warm = [w for w in self.streaming if map_id in w.maps]
            worker = min(warm or self.streaming, key=lambda w: len(w.keys))
            worker.keys.add(key)
            worker.maps.add(map_id)
            self.placed[key] = worker
            return worker

    def release(self, key: str) -> None:
        with self.lock:
            worker = self.placed.pop(key, None)
            if worker is not None:
                worker.keys.discard(key)

    def alive(self) -> bool:
        return all(w.process.is_alive() for w in self.streaming)
