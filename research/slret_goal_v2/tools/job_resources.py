"""Track only descendants of one job, even when torchrun workers use setsid."""
import os
from pathlib import Path
import subprocess


def process_table():
    result = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path/'stat').read_text().rsplit(')',1)[1].split()
            result[int(path.name)] = (int(fields[1]),int(fields[19]))  # PPID, start ticks
        except (FileNotFoundError,ProcessLookupError):
            pass
    return result


class ProcessTree:
    def __init__(self, root):
        self.root = root
        self.identities = {root:process_table()[root][1]}

    def members(self):
        table = process_table()
        owned = {pid for pid,start in self.identities.items()
                 if pid in table and table[pid][1] == start}
        while True:
            expanded = owned | {pid for pid,(parent,_) in table.items() if parent in owned}
            if expanded == owned:
                break
            owned = expanded
        self.identities.update({pid:table[pid][1] for pid in owned})
        return owned

    def gpu_bytes(self, timeout=5):
        owned = self.members()
        rows = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory',
            '--format=csv,noheader,nounits'],text=True,timeout=timeout)
        total = 0
        for row in rows.splitlines():
            if not row.strip():
                continue
            pid,mib = row.split(',')
            if int(pid) in owned:
                total += int(mib)*1024**2  # Fail closed on unavailable telemetry.
        return total

    def send_signal(self, sig):
        owned = self.members()
        for pid in sorted(owned,key=lambda pid:pid == self.root):
            try:
                current = process_table().get(pid)
                if current is not None and current[1] == self.identities[pid]:
                    os.kill(pid,sig)
            except ProcessLookupError:
                pass
