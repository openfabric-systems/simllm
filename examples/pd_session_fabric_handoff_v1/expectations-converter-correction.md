# Converter provenance correction

The expectations-only freeze named a converter with SHA-256
`df348be0f3e9138b2bf2e1f360df0a8c377f07161f2e16940e47d68bf9a3c50a`.
That 46,360-byte executable crashes in `Graph::serialize_mmap` on every tested
GOAL input, including an accepted historical two-rank trace. It can therefore
produce no valid binary artifact and cannot execute the frozen study.

Before the first scored packet run, the converter identity is corrected to the
tracked `htsim/sim/lgs/txt2bin` executable at the already frozen htsim gitlink
`1dcbfec36a33753bf978cf6323bade1a6645fe4f`. Its SHA-256 is
`f3745f34ad86febe9c9eebef10aee5fae00b8865cb29943344fb75b0f142495b`.
It converts the same historical trace successfully. The source gitlink,
packet backend binary, profile, sweep, PCIe term, endpoint pairing, exact
relations, physical bounds and fatal guards do not change.

Packet implementation edits existed in the uncommitted worktree when the
broken artifact was discovered. This correction commit contains no packet
implementation or scored result, and it precedes the implementation commit
and every scored run. It is a provenance repair, not a new behavioral freeze.
