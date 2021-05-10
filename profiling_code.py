import pstats
import io

pr = cProfile.Profile()
pr.enable()

num_stable_webs, num_converged = run_multiple(10,0.5,1)
pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps.print_stats()

with open('test.txt', 'w+') as f:
    f.write(s.getvalue())