"""
Utility functions and helpers.
"""

import os
import sys
from pathlib import Path
from datetime import datetime


class TeeOutput:
    """Redirects output to both a file and original stdout."""

    def __init__(self, file_handle, original_stdout):
        self.file = file_handle
        self.stdout = original_stdout

    def write(self, message):
        self.file.write(message)
        self.file.flush()
        self.stdout.write(message)
        self.stdout.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()


class Logger:
    """
    Simple logger that writes to both file and console.
    Auto-creates logs directory and increments log file numbers.
    Captures ALL stdout (including print statements from any module).
    """

    def __init__(self, log_dir="logs", prefix="run", console_echo=True, capture_stdout=True, log_path=None):
        """
        Initialize logger with auto-incremented log file or specific path.

        Args:
            log_dir (str): Directory for log files (default: "logs")
            prefix (str): Prefix for log file names (default: "run")
            console_echo (bool): Also print to console (default: True)
            capture_stdout (bool): Redirect all print() statements to log (default: True)
            log_path (Path or str): Specific log file path (overrides log_dir/prefix)
        """
        self.console_echo = console_echo
        self.capture_stdout = capture_stdout
        self.original_stdout = None
        self.tee_output = None

        if log_path:
            # Use specific log path
            self.log_path = Path(log_path)
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            log_filename = self.log_path.name
        else:
            # Auto-generate log path
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(exist_ok=True)

            # Find next available log number
            log_number = self._get_next_log_number(prefix)

            # Create log file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"{prefix}_{log_number:03d}_{timestamp}.log"
            self.log_path = self.log_dir / log_filename

        # Open log file
        self.log_file = open(self.log_path, 'w', buffering=1)  # Line buffered

        # Redirect stdout if requested
        if self.capture_stdout:
            self.original_stdout = sys.stdout
            self.tee_output = TeeOutput(self.log_file, self.original_stdout)
            sys.stdout = self.tee_output

        # Write header
        self._write_direct(f"=" * 60)
        self._write_direct(f"Log file: {log_filename}")
        self._write_direct(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._write_direct(f"=" * 60)
        self._write_direct("")

    def _get_next_log_number(self, prefix):
        """Find the next available log number."""
        existing_logs = list(self.log_dir.glob(f"{prefix}_*.log"))

        if not existing_logs:
            return 1

        # Extract numbers from existing log files
        numbers = []
        for log_file in existing_logs:
            try:
                # Parse: prefix_NNN_timestamp.log
                parts = log_file.stem.split('_')
                if len(parts) >= 2:
                    numbers.append(int(parts[1]))
            except (ValueError, IndexError):
                continue

        return max(numbers) + 1 if numbers else 1

    def _write_direct(self, message):
        """Write directly to both console and file without going through stdout redirection."""
        if self.original_stdout:
            self.original_stdout.write(str(message) + "\n")
            self.original_stdout.flush()
        else:
            print(message)
        self.log_file.write(str(message) + "\n")
        self.log_file.flush()

    def log(self, message=""):
        """
        Write message to log file and console.
        If stdout capture is enabled, just use print (it's already captured).
        Otherwise, write directly.

        Args:
            message (str): Message to log
        """
        if self.capture_stdout:
            # stdout is already redirected, just print normally
            print(message)
        else:
            # No redirection, write directly
            self._write_direct(message)

    def close(self):
        """Close the log file and restore stdout."""
        # Restore stdout before writing final messages
        if self.capture_stdout and self.original_stdout:
            sys.stdout = self.original_stdout

        # Write footer
        self._write_direct("")
        self._write_direct("=" * 60)
        self._write_direct(f"Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._write_direct("=" * 60)

        self.log_file.close()

    def __enter__(self):
        """Support context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up when used as context manager."""
        self.close()


def format_time(seconds):
    """
    Format seconds into human-readable time.

    Args:
        seconds (float): Time in seconds

    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


def save_individual(individual, filepath):
    """
    Save individual to file.

    Args:
        individual (Individual): Individual to save
        filepath (str): Path to save file
    """
    pass


def load_individual(filepath):
    """
    Load individual from file.

    Args:
        filepath (str): Path to load from

    Returns:
        Individual: Loaded individual
    """
    pass
