"""
journal.py -- append-only logging and atomic state files.

CONCEPT (why JSONL, not a database, in v0.1):
  * One JSON object per line, appended and never rewritten. That gives you a tamper-evident
    history: git shows exactly what was added and when.
  * Text diffs cleanly in git (SQLite is a binary file and would corrupt/conflict when a
    cloud job commits it every 6 hours).
  * A SQL database can be BUILT FROM these files later (see SPEC_v0.1.md section 10). The
    files stay the source of truth.

State that must be *replaced* (portfolio balances, current theses) is written atomically:
write to a temp file, then os.replace() -- so a crash mid-write can never leave a half-written
state file.
"""
import json
import os
from datetime import datetime, timezone

TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def now_utc():
    return datetime.now(timezone.utc)


def iso(dt):
    return dt.astimezone(timezone.utc).strftime(TS_FORMAT)


def parse_iso(s):
    return datetime.strptime(s, TS_FORMAT).replace(tzinfo=timezone.utc)


class Journal:
    def __init__(self, state_dir):
        self.dir = state_dir
        os.makedirs(os.path.join(state_dir, "arms"), exist_ok=True)

    def path(self, name):
        return os.path.join(self.dir, name)

    def append(self, name, row):
        line = json.dumps(row, sort_keys=True, default=str, separators=(",", ":"))
        with open(self.path(name), "a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def read(self, name):
        p = self.path(name)
        if not os.path.exists(p):
            return []
        with open(p, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def save_json(self, name, obj):
        final = self.path(name)
        tmp = final + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(obj, f, indent=2, sort_keys=True, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, final)

    def load_json(self, name, default=None):
        p = self.path(name)
        if not os.path.exists(p):
            return default
        with open(p, encoding="utf-8") as f:
            return json.load(f)
