"""Non-blocking terminal keyboard input (Windows and POSIX).

The OpenCV debug window handles keys when it has focus. This reader also lets
the keyboard controls work when the terminal has focus, and is the sole input
path for ``--no-preview``. It is disabled automatically for non-TTY stdin.
"""

from __future__ import annotations

import os
import platform
import sys
from collections import deque


class TerminalKeys:
    def __init__(self) -> None:
        self.enabled = False
        self._queue: deque[str] = deque(maxlen=64)
        self._saved_attrs = None
        self._fd = None

    def start(self) -> None:
        try:
            if not sys.stdin or not sys.stdin.isatty():
                return
            if platform.system() == "Windows":
                import msvcrt  # noqa: F401
            else:
                import termios
                import tty
                self._fd = sys.stdin.fileno()
                self._saved_attrs = termios.tcgetattr(self._fd)
                tty.setcbreak(self._fd)
            self.enabled = True
        except Exception as exc:
            print(f"[input] Terminal key input unavailable: {exc}", file=sys.stderr)
            self.enabled = False

    def poll(self) -> list[str]:
        if not self.enabled:
            return []
        try:
            if platform.system() == "Windows":
                import msvcrt
                while msvcrt.kbhit():
                    char = msvcrt.getwch()
                    if char in ("\x00", "\xe0"):
                        # Consume the second byte of function/arrow keys.
                        if msvcrt.kbhit():
                            msvcrt.getwch()
                        continue
                    self._push(char)
            else:
                import select
                while True:
                    ready, _, _ = select.select([sys.stdin], [], [], 0.0)
                    if not ready:
                        break
                    char = os.read(sys.stdin.fileno(), 1).decode(
                        sys.stdin.encoding or "utf-8", errors="ignore"
                    )
                    if not char:
                        break
                    if char == "\x1b":
                        # Consume any remainder of an arrow/function-key escape
                        # sequence so it cannot accidentally trigger A/P.
                        while select.select([sys.stdin], [], [], 0.002)[0]:
                            if not os.read(sys.stdin.fileno(), 1):
                                break
                        self._queue.append("x")
                        continue
                    self._push(char)
        except (OSError, ValueError):
            self.enabled = False
        return self.drain()

    def _push(self, char: str) -> None:
        if not char or char in ("\r", "\n", "\t"):
            return
        if char == "\x1b":
            # Escape cancels, but arrow-key sequences should not trigger it.
            self._queue.append("x")
            return
        self._queue.append(char.lower())

    def drain(self) -> list[str]:
        keys: list[str] = []
        while self._queue:
            keys.append(self._queue.popleft())
        return keys

    def close(self) -> None:
        if self._saved_attrs is not None and self._fd is not None:
            try:
                import termios
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved_attrs)
            except Exception:
                pass
        self._saved_attrs = None
        self.enabled = False
