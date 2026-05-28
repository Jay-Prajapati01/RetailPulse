"""Pages package for the dashboard.

This module intentionally does not import individual page modules to avoid
eagerly loading heavy dependencies at package import time. Page modules are
imported dynamically by the application when the user navigates to them.
"""

__all__ = []
