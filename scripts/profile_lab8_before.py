import cProfile, io, pstats, subprocess, sys
from pathlib import Path
from findex.index import build_inverted_index
from findex.search import clear_query_cache, search
from findex.storage import load_index, save_index

C, I, D, Q = Path("data/corpus_bench"), Path("data/index.bin"), Path("docs"), "whale OR sea OR captain"
D.mkdir(exist_ok=True)
if len(sys.argv) > 1 and sys.argv[1] == "search":
    idx = load_index(I)
    [clear_query_cache() or search(idx, Q, top_k=10) for _ in range(150)]
    sys.exit(0)
pr = cProfile.Profile(); pr.enable()
idx = build_inverted_index(C, executor="serial"); save_index(idx, Path("data/tmp_prof.bin"))
pr.disable(); s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(10)
print("=== cProfile index ===\n" + s.getvalue())
idx = load_index(I if I.exists() else Path("data/tmp_prof.bin"))
pr2 = cProfile.Profile(); pr2.enable()
[clear_query_cache() or search(idx, Q, top_k=10) for _ in range(100)]
pr2.disable(); s2 = io.StringIO(); pstats.Stats(pr2, stream=s2).sort_stats("tottime").print_stats(10)
print("=== cProfile search ===\n" + s2.getvalue())
ps = Path(sys.executable).parent / "py-spy.exe"
if ps.exists():
    subprocess.run([str(ps), "record", "-r", "250", "-o", "docs/flamegraph_index.svg", "--", sys.executable, "-m", "findex", "index", str(C), "--out", "data/tmp_prof.bin", "--executor", "serial"])
    c = "from pathlib import Path;from findex.storage import load_index;from findex.search import search,clear_query_cache;i=load_index(Path('data/index.bin'));[clear_query_cache() or search(i,'whale OR sea OR captain',top_k=10) for _ in range(250)]"
    subprocess.run([str(ps), "record", "-r", "250", "-o", "docs/flamegraph_search.svg", "--", sys.executable, "-c", c])
print("SVGs:", list(D.glob("*.svg")))
