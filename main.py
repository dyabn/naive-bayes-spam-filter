"""Application entry point."""

from __future__ import annotations

import tkinter as tk

import database
from ui import SpamMailApp


def main() -> None:
    database.initialize_database()

    root = tk.Tk()
    SpamMailApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
