.DEFAULT_GOAL := help

TEST_COMMAND = nosetests
help:
	@echo 'Use "make test" to run all the unit tests and docstring tests.'
	@echo 'Use "make pep8" to validate PEP8 compliance.'
	@echo 'Use "make html" to create html documentation with sphinx'
	@echo 'Use "make all" to run all the targets listed above.'
test:
	$(TEST_COMMAND)
pep8:
	pycodestyle mmvec setup.py
	flake8 mmvec setup.py

all: pep8 test
