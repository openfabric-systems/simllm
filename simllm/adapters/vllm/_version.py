"""Single source for the vLLM version pin.

Both the package ``__init__`` and the executor module need the pin, but the
package must stay lazily importable (it must not pull the executor module in
just to learn the version), so the constant lives here.
"""

#: vLLM release this adapter is written against and tested for API shape.
PINNED_VLLM_VERSION = "0.27.1"
