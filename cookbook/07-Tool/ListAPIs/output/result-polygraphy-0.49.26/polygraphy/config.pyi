from __future__ import annotations
import os as os
import sys as sys
__all__: list[str] = ['ARRAY_SWAP_THRESHOLD_MB', 'ASK_BEFORE_INSTALL', 'AUTOINSTALL_DEPS', 'INSTALL_CMD', 'INTERNAL_CORRECTNESS_CHECKS', 'USE_TENSORRT_RTX', 'os', 'sys']
ARRAY_SWAP_THRESHOLD_MB: int = -1
ASK_BEFORE_INSTALL: bool = False
AUTOINSTALL_DEPS: bool = False
INSTALL_CMD: list = ['/usr/bin/python3', '-m', 'pip', 'install']
INTERNAL_CORRECTNESS_CHECKS: bool = False
USE_TENSORRT_RTX: bool = False
