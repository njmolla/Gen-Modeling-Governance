import pstats
import io
import cProfile

pr = cProfile.Profile()
pr.enable()

np.random.seed(0)
num_stable_webs, num_converged = run_multiple(10,0.5,1)
pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps.print_stats()

with open('profiling_results.txt', 'w+') as f:
    f.write(s.getvalue())