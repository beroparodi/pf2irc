"""Runs the disc-burst models: python3 runall.py [names]  (default: all six; two at a time)."""
import sys
from multiprocessing import Pool
from discburst import run, RUNS
def go(n): run(n, **RUNS[n])
if __name__=="__main__":
    names=sys.argv[1:] or list(RUNS)
    with Pool(2) as p: p.map(go,names,chunksize=1)
