"""Single source for the SGLang pin.

SGLang moves fast and cuts releases rarely relative to its main branch, so
the adapter pins the commit it was written and verified against rather than
a version string. Both the package ``__init__`` and the worker module need
the pin, and the package must stay lazily importable, so the constant lives
here.
"""

#: SGLang main-branch commit this adapter is written against (2026-08-24).
PINNED_SGLANG_COMMIT = "bfeae4e"
