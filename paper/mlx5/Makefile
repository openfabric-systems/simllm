# Build the mlx5 reverse-engineering paper.
#
#   make            build main.pdf (figures first)
#   make figures    regenerate the plot PDFs from data/*.csv
#   make clean      remove LaTeX auxiliary files
#   make distclean  also remove main.pdf and the generated figures
#
# Requirements: TeX Live with the acmart class (tlmgr install acmart) and,
# for the figures, Python 3 with matplotlib.

PYTHON  ?= python3
LATEXMK ?= latexmk

FIGURES := figures/msgsize.pdf figures/drain.pdf figures/incast.pdf

.PHONY: all figures clean distclean

all: main.pdf

main.pdf: main.tex refs.bib $(wildcard sections/*.tex) figures/arch.tex $(FIGURES)
	$(LATEXMK) -pdf main.tex

figures: $(FIGURES)

figures/msgsize.pdf: tools/plot_msgsize.py data/msgsize.csv
	$(PYTHON) tools/plot_msgsize.py --data data/msgsize.csv --out $@

figures/drain.pdf: tools/plot_drain.py data/gapsweep.csv
	$(PYTHON) tools/plot_drain.py --data data/gapsweep.csv --out $@

figures/incast.pdf: tools/plot_incast.py data/incast.csv data/buffer.csv
	$(PYTHON) tools/plot_incast.py --data data/incast.csv \
		--buffer data/buffer.csv --out $@

clean:
	$(LATEXMK) -c

distclean:
	$(LATEXMK) -C
	rm -f $(FIGURES)
