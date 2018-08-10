SHELL=/bin/bash
MAKE=make --no-print-directory

install:
	python setup.py install --user

test:
	python -m unittest discover ./mlmisc/test

uninstall:
	pip uninstall mlmisc

.PHONY:
	install uninstall
