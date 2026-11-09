# H2O Setup

From this folder, run:

```sh
./run_h2o.sh
```

To set up on another computer, install Python 3.14 and Java 17, then run:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements_h2o.txt
./run_h2o.sh
```

Keep `athletes.csv` in this folder. Results are saved in `h2o_results/` and `H2O_Analysis.md`.
