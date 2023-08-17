#!/bin/bash

# tmp skripti tähän ajamiseen ettei tarvii koko ajan koneella kökkiä
eval "$(conda shell.bash hook)"
conda activate xgb

python get-swi-ts.py 10000 20000 5000 10000
python get-swi-ts.py 20000 30000 10000 15000
python get-swi-ts.py 30000 40000 15000 20000
python get-swi-ts.py 40000 50000 20000 25000
python get-swi-ts.py 50000 60000 25000 30000
python get-swi-ts.py 60000 70000 30000 35000
python get-swi-ts.py 70000 80000 35000 40000
python get-swi-ts.py 80000 90000 40000 45000
python get-swi-ts.py 90000 100000 45000 50000
python get-swi-ts.py 100000 110000 50000 55000
python get-swi-ts.py 110000 120000 55000 60000
