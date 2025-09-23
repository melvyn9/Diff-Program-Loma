import subprocess, time

def run(mode, loma_file, binary):
    print(f"\n=== {mode.upper()} MODE ===")
    # 1) regenerate d_func.c and mpi_main.c
    subprocess.run(
      ["python", "compiler.py", loma_file,
       "--diff_mode", mode, "--target", "c", "--out", "d_func.c"],
      check=True
    )
    # 2) compile the ONE mpi_main.c
    subprocess.run(
      ["mpicc", "mpi_main.c", "d_func.c", "-lm", "-o", binary],
      check=True
    )

    for np in [1, 2, 4, 6, 8]:
        print(f"\nRunning with {np} processes:")
        cmd = ["mpirun", "-np", str(np), binary]
        if np > 4:
            cmd.insert(1, "--oversubscribe")
        start = time.perf_counter()
        subprocess.run(cmd, check=True)
        print(f"Time: {time.perf_counter() - start:.4f} sec")

run("fwd", "benchmark_3d.loma",      "./dparallel_bench_fwd")
run("rev", "benchmark_3d_rev.loma", "./dparallel_bench_rev")
