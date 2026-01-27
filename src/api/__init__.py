"""
FastAPI Dashboard Module

Provides a REST API and web dashboard for the trading system.
"""

from .main import create_app, run_server

__all__ = ["create_app", "run_server"]
