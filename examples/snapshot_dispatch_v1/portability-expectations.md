# Runtime identity portability amendment

This expectations-only amendment responds to a Windows continuous-integration
failure: the `math` module is built into that Python runtime and has no
separate `__file__`. The original Linux campaign and its source receipts remain
unchanged. This portability requirement is post-specified to the observed
software failure; the fixture grid below precedes the correction and its runs.

Keep every file-backed library identity exactly as before. For a module
without a file, require an actual built-in module name and built-in import
origin. On Windows, record the path and hash of the loaded Python dynamic-link
library, together with the explicit built-in origin. Python exposes the loaded
library handle as [`sys.dllhandle`](https://docs.python.org/3.12/library/sys.html#sys.dllhandle).
Windows resolves that handle to its actual file through
[`GetModuleFileNameW`](https://learn.microsoft.com/en-us/windows/win32/api/libloaderapi/nf-libloaderapi-getmodulefilenamew).
Do not guess a filename, substitute a launcher executable, omit the module or
weaken the source comparison when no file is present.

Before any new fixture run, freeze a two-by-two grid: file-backed or verified
built-in module, crossed with two distinct backing-file contents. File-backed
rows retain their existing exact fields and encoding. Every row binds the
selected file bytes; changing those bytes changes the hash. Built-in rows name
the built-in origin and the actual runtime image. Missing files, unknown module
names, unrecognized origins, absent or invalid runtime handles, operating-system
lookup failure and truncated paths reject. A real Windows runtime identity
must include a nonempty image hash for its built-in math module.

These are unscored evidence-conformance and rejection checks. They change no
snapshot semantics, engine behavior, model price, timing sweep or resource
limit. Existing accepted Linux native and snapshot records keep their actual
executed source identifiers. The model campaign is not rerun or rescored by
this reader correction. CORE-68 closure still requires its independent native
review and all software gates.
