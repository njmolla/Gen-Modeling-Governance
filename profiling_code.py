import pstats
import io
import cProfile
from GM_code import run_once
pr = cProfile.Profile()
pr.enable()

N1 = 3
N2 = 2
N3 = 3
K = 0
M = 1
T = sum(N1,N2,N3,K,M)
C = 0.5
num_stable_webs, num_converged = run_once(N1,N2,N3,K,M,T,C,sample_exp=True)
pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps.print_stats()

with open('profiling_results.txt', 'w+') as f:
    f.write(s.getvalue())