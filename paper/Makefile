.PHONY: pdf clean distclean

pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error main.tex

clean:
	latexmk -c

distclean: clean
	rm -f main.pdf
