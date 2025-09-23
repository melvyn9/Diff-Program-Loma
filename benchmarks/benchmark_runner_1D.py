import subprocess, time

def run(mode, loma_file, c_main, binary):
    print(f"\n=== {mode.upper()} MODE ===")
    subprocess.run(["python", "compiler.py", loma_file, "--diff_mode", mode, "--target", "c", "--out", "d_func.c"])
    subprocess.run(["mpicc", c_main, "d_func.c", "-lm", "-o", binary])

    for np in [1, 2, 4, 6, 8]:
        print(f"\nRunning with {np} processes:")
        cmd = ["mpirun", "-np", str(np), binary]
        if np > 4:
            cmd.insert(1, "--oversubscribe")
        start = time.perf_counter()
        subprocess.run(cmd)
        print(f"Time: {time.perf_counter() - start:.4f} sec")

run("fwd", "benchmark_3d.loma", "mpi_main_fwd.c", "./dparallel_bench_fwd")
run("rev", "benchmark_3d_rev.loma", "mpi_main_rev.c", "./dparallel_bench_rev")
