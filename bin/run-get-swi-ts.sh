#!/bin/bash

# tmp skripti tähän ajamiseen ettei tarvii koko ajan koneella kökkiä
eval "$(conda shell.bash hook)"
conda activate xgb

python get-swi-ts-all.py 25000 30000
python get-swi-ts-all.py 30000 35000
python get-swi-ts-all.py 35000 40000
python get-swi-ts-all.py 40000 45000
python get-swi-ts-all.py 45000 50000
python get-swi-ts-all.py 50000 55000
python get-swi-ts-all.py 55000 60000