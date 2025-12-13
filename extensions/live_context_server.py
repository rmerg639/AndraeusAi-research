"""
Live Context Server
Real-time file tracking for dynamic AI personalization.

This creates a HYBRID system:
- Static knowledge: Fine-tuned facts (name, birthday, protocols) - always available
- Dynamic context: Recent file changes injected at runtime - always current

Architecture:
1. File watcher monitors specified directories
2. On change, updates context cache
3. Before each inference, injects relevant file context into prompt
4. AI sees both trained knowledge AND current file state

Usage:
    python live_context_server.py --watch "C:/Projects" --port 8080

Then query your personal AI through this server instead of directly.

Copyright (c) 2025 Rocco Andraeus Sergi. All Rights Reserved.
Licensed under Andraeus AI Proprietary License v2.2
"""

import os
import time
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from threading import Thread, Lock
import argparse

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    "watch_paths": [],
    "watch_extensions": [".py", ".md", ".txt", ".json", ".yaml", ".yml"],
    "ignore_patterns": ["__pycache__", ".git", "node_modules", ".venv", "*.pyc"],
    "max_context_files": 5,
    "max_chars_per_file": 2000,
    "refresh_interval_seconds": 10,
    "context_window_hours": 24,  # Only include files modified in last N hours
}


@dataclass
class FileContext:
    """Represents a tracked file's context."""
    path: str
    content: str
    modified_time: float
    hash: str
    size: int


@dataclass
class ContextCache:
    """Cache of recently modified files for context injection."""
    files: Dict[str, FileContext] = field(default_factory=dict)
    last_scan: float = 0
    lock: Lock = field(default_factory=Lock)


# =============================================================================
# FILE WATCHER
# =============================================================================

class LiveContextWatcher:
    """Watches directories for file changes and maintains context cache."""

    def __init__(self, config: dict = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.cache = ContextCache()
        self.running = False
        self._watch_thread: Optional[Thread] = None

    def should_ignore(self, path: str) -> bool:
        """Check if path matches ignore patterns."""
        path_str = str(path)
        for pattern in self.config["ignore_patterns"]:
            if pattern.startswith("*"):
                if path_str.endswith(pattern[1:]):
                    return True
            elif pattern in path_str:
                return True
        return False

    def should_include(self, path: str) -> bool:
        """Check if file extension is watched."""
        return Path(path).suffix.lower() in self.config["watch_extensions"]

    def get_file_hash(self, content: str) -> str:
        """Generate hash of file content."""
        return hashlib.md5(content.encode()).hexdigest()[:8]

    def scan_directory(self, dir_path: str) -> List[FileContext]:
        """Scan a directory for relevant files."""
        results = []
        cutoff_time = time.time() - (self.config["context_window_hours"] * 3600)

        try:
            for root, dirs, files in os.walk(dir_path):
                # Filter out ignored directories
                dirs[:] = [d for d in dirs if not self.should_ignore(os.path.join(root, d))]

                for file in files:
                    file_path = os.path.join(root, file)

                    if self.should_ignore(file_path):
                        continue

                    if not self.should_include(file_path):
                        continue

                    try:
                        stat = os.stat(file_path)
                        if stat.st_mtime < cutoff_time:
                            continue  # File too old

                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read(self.config["max_chars_per_file"])

                        results.append(FileContext(
                            path=file_path,
                            content=content,
                            modified_time=stat.st_mtime,
                            hash=self.get_file_hash(content),
                            size=stat.st_size
                        ))
                    except (IOError, OSError):
                        continue

        except (IOError, OSError) as e:
            print(f"Error scanning {dir_path}: {e}")

        return results

    def refresh_cache(self):
        """Scan all watch paths and update cache."""
        all_files = []

        for watch_path in self.config["watch_paths"]:
            if os.path.exists(watch_path):
                all_files.extend(self.scan_directory(watch_path))

        # Sort by modification time, newest first
        all_files.sort(key=lambda f: f.modified_time, reverse=True)

        # Keep only top N files
        top_files = all_files[:self.config["max_context_files"]]

        with self.cache.lock:
            self.cache.files = {f.path: f for f in top_files}
            self.cache.last_scan = time.time()

    def get_context_string(self) -> str:
        """Generate context string for injection into prompt."""
        with self.cache.lock:
            if not self.cache.files:
                return ""

            lines = ["\n--- LIVE FILE CONTEXT ---"]
            for path, ctx in self.cache.files.items():
                rel_path = os.path.basename(path)
                mod_time = datetime.fromtimestamp(ctx.modified_time).strftime("%H:%M")
                lines.append(f"\n[{rel_path}] (modified {mod_time}):")
                lines.append(ctx.content[:500] + ("..." if len(ctx.content) > 500 else ""))

            lines.append("\n--- END LIVE CONTEXT ---")
            return "\n".join(lines)

    def _watch_loop(self):
        """Background thread for periodic scanning."""
        while self.running:
            self.refresh_cache()
            time.sleep(self.config["refresh_interval_seconds"])

    def start(self):
        """Start background file watching."""
        if self.running:
            return

        self.running = True
        self.refresh_cache()  # Initial scan

        self._watch_thread = Thread(target=self._watch_loop, daemon=True)
        self._watch_thread.start()
        print(f"Live context watcher started. Monitoring {len(self.config['watch_paths'])} paths.")

    def stop(self):
        """Stop background file watching."""
        self.running = False
        if self._watch_thread:
            self._watch_thread.join(timeout=2)


# =============================================================================
# CONTEXT-AWARE INFERENCE
# =============================================================================

def inject_live_context(system_prompt: str, watcher: LiveContextWatcher) -> str:
    """Inject live file context into system prompt."""
    context = watcher.get_context_string()
    if context:
        return system_prompt + context
    return system_prompt


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def demo():
    """Demonstrate live context tracking."""
    config = {
        "watch_paths": [
            os.path.expanduser("~/Documents"),
            os.path.expanduser("~/Projects"),
        ],
        "watch_extensions": [".py", ".md", ".txt"],
        "max_context_files": 3,
        "refresh_interval_seconds": 5,
    }

    watcher = LiveContextWatcher(config)
    watcher.start()

    print("\nLive Context Watcher Demo")
    print("=" * 50)
    print(f"Watching: {config['watch_paths']}")
    print(f"Extensions: {config['watch_extensions']}")
    print("\nPress Ctrl+C to stop\n")

    try:
        while True:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Current context:")
            print("-" * 40)
            context = watcher.get_context_string()
            if context:
                print(context[:500] + "..." if len(context) > 500 else context)
            else:
                print("No recent files in watched paths.")
            time.sleep(10)
    except KeyboardInterrupt:
        print("\nStopping watcher...")
        watcher.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Context Server for Personal AI")
    parser.add_argument("--watch", nargs="+", help="Directories to watch")
    parser.add_argument("--demo", action="store_true", help="Run demo mode")

    args = parser.parse_args()

    if args.demo:
        demo()
    elif args.watch:
        config = {"watch_paths": args.watch}
        watcher = LiveContextWatcher(config)
        watcher.start()

        print(f"Watching: {args.watch}")
        print("Press Ctrl+C to stop")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            watcher.stop()
    else:
        print("Usage: python live_context_server.py --demo")
        print("       python live_context_server.py --watch /path/to/watch")
