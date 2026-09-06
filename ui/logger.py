import io
import sys
from typing import List


class TerminalLogger(io.StringIO):
    def __init__(self, max_lines: int = 50):
        super().__init__()
        self.max_lines = max_lines
        self.log_lines: List[str] = []
        self._orig_stdout = sys.stdout

    def write(self, s: str):
        if self._orig_stdout:
            self._orig_stdout.write(s)
            self._orig_stdout.flush()

        if s.strip('\r\n'):
            for line in s.splitlines():
                clean_line = line.strip()
                if clean_line:
                    self.log_lines.append(clean_line)
                    if len(self.log_lines) > self.max_lines:
                        self.log_lines.pop(0)

    def flush(self):
        if self._orig_stdout:
            self._orig_stdout.flush()

    def start_capture(self):
        sys.stdout = self

    def restore(self):
        sys.stdout = self._orig_stdout
