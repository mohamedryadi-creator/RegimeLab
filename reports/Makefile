# Build the RegimeLab research report.
#
# Requires Tectonic (https://tectonic-typesetting.github.io):
#   brew install tectonic
# Fallback without Tectonic: run `pdflatex report.tex` twice.

report.pdf: report.tex $(wildcard figures/*.png)
	tectonic report.tex

.PHONY: clean
clean:
	rm -f report.pdf report.aux report.log report.out report.toc report.synctex.gz
