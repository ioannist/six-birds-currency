.PHONY: pdf clean distclean flatten

pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error main.tex
	$(MAKE) flatten

flatten:
	mkdir -p build
	@if command -v latexpand >/dev/null 2>&1; then \
		latexpand -o build/main_flat.tex main.tex; \
	else \
		echo "latexpand not found; cannot write build/main_flat.tex" >&2; \
		exit 1; \
	fi

clean:
	latexmk -c

distclean: clean
	rm -f main.pdf
