"""Portable log directory resolution (works on any machine / install layout)."""

from __future__ import annotations

import glob
import os
from typing import List, Optional


def resolve_log_directory(configured: str = '', caller_file: str | None = None) -> str:
    """
    Resolve where CSV logs are written.

    Priority:
      1. DOCKING_LOG_DIRECTORY environment variable
      2. ROS parameter log_directory (if non-empty)
      3. <package_root>/logs next to the installed or source package
    """
    env = os.environ.get('DOCKING_LOG_DIRECTORY', '').strip()
    if env:
        return os.path.abspath(os.path.expanduser(env))

    configured = (configured or '').strip()
    if configured:
        return os.path.abspath(os.path.expanduser(configured))

    script_dir = os.path.dirname(os.path.realpath(caller_file or __file__))
    if os.path.basename(script_dir) == 'sdc_core':
        package_root = os.path.dirname(script_dir)
    elif os.path.basename(os.path.dirname(script_dir)) == 'lib':
        package_root = os.path.dirname(os.path.dirname(script_dir))
    else:
        package_root = os.path.dirname(script_dir)

    return os.path.join(package_root, 'logs')


def log_search_directories(configured: str = '', caller_file: str | None = None) -> List[str]:
    """Directories to search for docking_*.csv (newest first candidate list)."""
    dirs = []
    env = os.environ.get('DOCKING_LOG_DIRECTORY', '').strip()
    if env:
        dirs.append(os.path.abspath(os.path.expanduser(env)))
    if configured and str(configured).strip():
        dirs.append(os.path.abspath(os.path.expanduser(str(configured).strip())))
    dirs.append(resolve_log_directory('', caller_file=caller_file))
    return list(dict.fromkeys(dirs))


def find_latest_log(configured: str = '', caller_file: str | None = None) -> Optional[str]:
    """Newest docking_*.csv across known log directories."""
    latest_path = None
    latest_mtime = -1.0
    for logs_dir in log_search_directories(configured, caller_file=caller_file):
        pattern = os.path.join(logs_dir, 'docking_*.csv')
        for path in glob.glob(pattern):
            mtime = os.path.getmtime(path)
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_path = path
    return latest_path
