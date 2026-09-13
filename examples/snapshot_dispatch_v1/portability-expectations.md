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

## POSIX runtime image amendment

This expectations-only amendment responds to a Linux interpreter whose `math`
module is built into the runtime and therefore carries no `__file__`, observed
on a python-build-standalone CPython 3.12 and on a Debian system Python. The
Windows amendment above and every accepted record stay unchanged. This POSIX
requirement is post-specified to the observed software failure; the fixture
grid below precedes the correction and its runs.

Keep every file-backed library identity and the Windows lookup byte identical.
For a module without a file on Linux, resolve the loaded image that contains
the interpreter core: take the address of the exported core symbol
`Py_Initialize` through
[`ctypes.pythonapi`](https://docs.python.org/3.12/library/ctypes.html#ctypes.pythonapi)
and locate that address inside the kernel's memory map of the running process,
the `self/maps` entry of the process filesystem, which records the mapped file
path of every region. On other POSIX platforms use the dynamic loader's own
record of that address
([`dladdr`](https://man7.org/linux/man-pages/man3/dladdr.3.html), field
`dli_fname`). Never guess a filename from `sys.executable`, a launcher or
`sysconfig`. Hash the resolved image bytes and record the explicit built-in
origin.

Before any new fixture run, reuse the frozen two-by-two grid: file-backed or
verified built-in module, crossed with two distinct backing-file contents.
File-backed rows retain their existing exact fields and encoding. Every row
binds the selected image bytes; changing those bytes changes the hash.
Built-in rows on Linux name the built-in origin and the actual mapped image. A
missing `pythonapi` core symbol, no mapping containing the symbol address, an
anonymous or pseudo mapping (an empty path or one in square brackets such as
`[vdso]`), a relative or missing image path under strict resolution, and a
loader lookup failure all reject. On the loader branch the same rejections
apply to its own evidence: a zero return or an empty loader name is a lookup
failure, and a relative or missing loader name rejects under the identical
strict resolution. A real Linux runtime identity must include a
nonempty image hash for its built-in `math` module whenever that module has no
file, and `runtime_before` must equal `runtime_after`.

These are unscored evidence-conformance and rejection checks. They change no
snapshot semantics, engine behavior, model price, timing sweep or resource
limit. Existing accepted records keep their identifiers. The model campaign is
not rerun or rescored by this reader correction.
