.PHONY: setup train evaluate bundle predict test

setup:
\tpython -m pip install --upgrade pip
\tpip install -r requirements.txt
\t@if exist requirements-dev.txt (pip install -r requirements-dev.txt)

train:
\tpython -m creditcard_psp.modeling.train

evaluate:
\tpython -m creditcard_psp.plots --out reports/figures

bundle:
\tpython -m creditcard_psp.modeling.predict --dump-bundle

predict:
\tpython -m creditcard_psp.modeling.predict --input data/sample_input.csv --out predictions.csv

test:
\tpytest -q
