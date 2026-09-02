# Reverse engineering the ConnectX-5 into a block architecture (paper draft)

First draft, v0.1. The paper describes how an mlx5-class RDMA NIC is
reconstructed from public information and black-box measurement into a
measurement-calibrated block architecture that can be implemented on an AMD
Alveo U55C, with the SimLLM golden C model as the reference implementation.
The block architecture is the centrepiece; the measurement sections summarise
the campaign rather than re-deriving it.

## Build

```bash
cd paper/mlx5
make            # figures first, then latexmk -pdf main.tex
```

Requirements:

- TeX Live with the `acmart` class. On a user-owned TeX Live install:

  ```bash
  tlmgr update --self
  tlmgr install acmart xstring libertine newtx inconsolata environ totpages \
      trimspaces comment ncctools hyperxmp ifmtarg mweights fontaxes \
      doclicense preprint cm-super binhex draftwatermark float ms zref \
      refcount pdftexcmds mathtools multirow
  ```

  `acmart` is installed as a TeX Live package, not vendored into this
  directory, so no class files are tracked here. If a build host cannot run
  `tlmgr`, the class is LPPL licensed and may be copied from CTAN into this
  directory instead; nothing else in the sources would change.

- Python 3 with `matplotlib`, only for `make figures`. The generated figure
  PDFs are tracked, so a plain `latexmk -pdf main.tex` builds the paper
  without Python.

`make clean` removes the LaTeX auxiliary files; `make distclean` also removes
`main.pdf` and the generated figures. Both the built PDF and the auxiliary
files are untracked through the local `.gitignore`.

## Class choice

The document class is named exactly once, on the `\documentclass` line at the
top of `main.tex`, and the file says so in a comment. Everything under
`sections/` is plain LaTeX plus a small macro layer defined in `main.tex`
(`\evdoc`, `\evdrv`, `\evcal`, `\evdecl`, `\evinf`, `\ctr`, `\csr`, `\anom`,
`\TODO` and four unit macros). No section file uses a class-specific command.

To switch to IEEEtran:

1. Replace `\documentclass[sigconf,nonacm]{acmart}` with
   `\documentclass[conference]{IEEEtran}`.
2. Drop the acmart-only front matter in `main.tex`: `\citestyle`,
   `\setcopyright`, `\settopmatter`, `\renewcommand\footnotetextcopyrightpermission`,
   `\acmConference`, and the `acks` environment (IEEEtran uses
   `\section*{Acknowledgment}`).
3. Replace `\affiliation{...}` and `\email{}` with an IEEEtran
   `\IEEEauthorblockN` / `\IEEEauthorblockA` pair, and
   `\bibliographystyle{ACM-Reference-Format}` with
   `\bibliographystyle{IEEEtran}`.
4. Add `\usepackage{booktabs,graphicx,xcolor,amsmath,hyperref}`, which acmart
   provides and IEEEtran does not.

Nothing under `sections/`, `figures/`, `data/` or `tools/` changes.

## Layout

```
main.tex                  class choice, macro layer, front matter, includes
sections/abstract.tex     abstract
sections/intro.tex        motivation and contributions
sections/legal.tex        sources used and the basis in United States law
sections/background.tex   RoCEv2 transport, the mlx5 interface, DCQCN
sections/measurement.tex  the campaign, the instruments and the findings
sections/anomaly-table.tex  the nineteen-row anomaly table (Table 3)
sections/architecture.tex target, interconnects, register map, expandability
sections/blocks.tex       the nineteen block subsections
sections/verification.tex golden model, DPI-C and UVM, the full chain
sections/related.tex      related work
sections/status.tex       status, roadmap and conclusion
figures/arch.tex          the top-level block diagram, pure TikZ
figures/*.pdf             generated plots (tracked)
data/*.csv                curated measurement extracts (tracked)
tools/plot_*.py           regenerate the plots from data/
refs.bib                  bibliography
Makefile, latexmkrc       build
```

## Where the numbers come from

Every measured number in the paper comes from the mlx5 campaign records, which
live in the companion measurement repository under
`report/mlx5-campaign/`. The paper cites them by record name:

| Record | What the paper takes from it |
|---|---|
| `RESULTS-p1-harness` | instrument hardening, host-loopback per-type ceilings, the PCIe pressure sweep |
| `RESULTS-p2-msgsize` | the message-size and latency sweeps, the first fixed-offset fit |
| `RESULTS-p3-collie` | the wire ceilings, the anomaly seeds, the in-NIC budget, the instrument defects |
| `RESULTS-p4-kernels` | the true depth-1 refit, the drain window, duplex against simplex, skew |
| `RESULTS-p5a-incast` | the two-to-one incast tax, fair share, fan-out, the unreliable one-over-N result |
| `RESULTS-p5b-tos0` | the on-wire type-of-service probe and the ECT(0) stamp |
| `RESULTS-p6-fabric` | the topology, the egress buffer, zero switch marking, DCQCN dynamics, the lone-flow floor, counter semantics |
| `FINDINGS-cx5` | the consolidated ledger and the graduating constants |
| `VALIDITY-lossy-fabric` | which tests a fabric without priority flow control can and cannot support |
| `data/p5a/congestion_control_config.md` | the endpoint congestion-control audit |

Model-side material comes from this repository:
`docs/design/rnic-cmodel.md` (the golden model design),
`docs/design/rnic-anomaly-table.md` (the nineteen rows, generated from the
model sources), `docs/papers/rnic-hardware-calibration.md` (the evidence-class
contract), `simllm/backends/rnic/` (the implementation) and the study records
under `examples/cx5_msgsize_v1`, `examples/hacc_fabric_v1`,
`examples/rnic_cmodel_v1`, `examples/rnic_cmodel_rx_v1` and
`examples/rnic_cmodel_cc_v1`.

### Curated data

`data/` holds four small extracts of the campaign CSVs, each with a comment
header naming its source record and its columns. Node names and addresses are
removed. The extracts are:

- `msgsize.csv`: goodput against message size for three arms (the benchmark
  depth-1 loop, the true depth-1 engine arm, and the depth-1024 engine arm).
- `gapsweep.csv`: goodput inside a burst, goodput over the wall clock, and the
  receiver's ingress discard delta, against the inter-burst gap.
- `incast.csv`: the two-to-one incast cases with their solo and fan-out
  controls.
- `buffer.csv`: the switch egress buffer identity, twelve runs at three excess
  rates.

The extraction from the full campaign CSVs was a one-off; `tools/` regenerates
the *plots* from these extracts, not the extracts from the raw data. Each
script takes `--data` (and `--buffer` for the incast plot) and `--out`, and
contains no absolute path.

## Open TODOs in the draft

Marked in red in the built PDF by the `\TODO` macro, and listed here so they
can be found without grepping:

1. `sections/architecture.tex`, Table 4: confirm every Alveo U55C resource
   figure against the current data sheet. The design was sized against 1.30 M
   lookup tables, about 70 Mb block RAM and 270 Mb ultra RAM, 16 GB of HBM2,
   two QSFP28 cages and PCIe Gen3 by 16 or Gen4 by 8. The brief for this draft
   quoted 8 GB of HBM2, which is the figure for a different Alveo card.
2. `sections/related.tex`: confirm the UCCL title, author list and publication
   status.
3. `sections/status.tex`: the architecture is a design and no
   register-transfer-level code exists yet. Replace the status paragraph with
   real resource and timing numbers once the first blocks build.
4. `main.tex`: the acknowledgements are a placeholder.
5. `refs.bib`: two entries carry a note recording that the brief's venue year
   differed from the published one (Coyote is OSDI 2020, not 2021; the FlexNIC
   ASPLOS paper is 2016, and the 2015 FlexNIC paper is the HotOS position
   paper). Confirm which the paper should cite.
6. Author list is `Yifeng Wang and collaborators, ETH Zurich`. No co-authors or
   affiliations were invented; fill in before submission.
7. `refs.bib`: no entry carries `address`, and three carry no page range.
   `ACM-Reference-Format` warns about each. The fields were left out rather
   than guessed; fill them from the publisher's page before submission. These
   are the only bibtex warnings in the build.
8. The draft is 20 pages. A SIGCOMM submission is 12 pages of body plus
   references, so about a third has to come out. The obvious candidates are
   the block subsections in `sections/blocks.tex`, which could become one
   dense table plus prose for the five blocks that carry the anomaly rows,
   and the background section, which could shrink to a paragraph.

## Remaining build warnings

`latexmk -pdf` finishes with no overfull or underfull horizontal boxes and no
LaTeX or hyperref warnings. Two classes of message remain:

- `Package balance Warning: You have called \balance in second column`. This
  comes from `acmart` balancing the columns of the last page and is benign; it
  moves or disappears whenever the content length changes.
- About fifteen `Underfull \vbox ... while \output is active`. These are the
  usual consequence of `\flushbottom` (which `acmart` `sigconf` requires) on
  a draft with many top-placed floats. They will settle as the content is
  trimmed to length.
