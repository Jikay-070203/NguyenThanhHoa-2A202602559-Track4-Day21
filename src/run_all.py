"""Chạy toàn bộ Topic A một lệnh: demo -> sweep -> detect -> failure -> latency -> B1/B2/B6 -> kiểm tra tái lập -> REPORT.md.

    python -m src.run_all --class-name "AI20K-xx"
    python -m src.run_all --max-frames 8 --max-nusc-frames 8 --skip-latency     # chạy thử nhanh

Mọi bước chỉ ghi vào --out-dir (mặc định results/) và --report-out (mặc định report/REPORT.md), không đụng vào data/.
"""
from __future__ import annotations

import argparse
import hashlib
import tempfile
import time
from pathlib import Path

from src import (exp_compare, exp_demo, exp_detect, exp_failure, exp_latency, exp_stress, exp_sweep, exp_synthetic,
                 make_report)


def determinism_check(out_dir: Path) -> str:
    """Chạy lại exp_sweep --quick hai lần, so sánh sha256 của CSV thô: cùng input phải ra cùng byte."""
    digests = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as d:
            exp_sweep.main(["--quick", "--no-plots", "--out-dir", d])
            digests.append(hashlib.sha256((Path(d) / "calib_sweep_frames.csv").read_bytes()).hexdigest())
    ok = digests[0] == digests[1]
    text = (f"{'PASS' if ok else 'FAIL'}: chạy lại `exp_sweep --quick` 2 lần cho cùng checksum "
            f"(sha256 {digests[0][:12]} / {digests[1][:12]})" if ok else
            f"FAIL: hai lần chạy `exp_sweep --quick` cho checksum khác nhau ({digests[0][:12]} / {digests[1][:12]})")
    (out_dir / "determinism_check.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return text


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Topic A - chay toan bo thi nghiem va sinh bao cao")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--report-out", default="report/REPORT.md")
    ap.add_argument("--class-name", default="[ĐIỀN lớp]")
    ap.add_argument("--student-name", default="Nguyen Thanh Hoa")
    ap.add_argument("--max-frames", type=int, default=0, help="gioi han so frame moi dataset cho sweep (0 = tat ca)")
    ap.add_argument("--max-nusc-frames", type=int, default=0, help="gioi han so frame nuScenes cho case Time (0 = tat ca)")
    ap.add_argument("--skip-latency", action="store_true")
    ap.add_argument("--skip-bonus", action="store_true", help="bo qua B1 (exp_compare), B2 (exp_stress), B6 (exp_synthetic)")
    ap.add_argument("--skip-determinism", action="store_true")
    ap.add_argument("--skip-report", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def step(name, fn):
        t0 = time.perf_counter()
        print(f"\n===== {name} =====", flush=True)
        fn()
        print(f"[{name}] xong sau {time.perf_counter() - t0:.0f}s", flush=True)

    od = ["--out-dir", str(out)]
    mf = ["--max-frames", str(args.max_frames)]
    step("1/10 demo + kiem tra tay CP2", lambda: exp_demo.main(od))
    step("2/10 sweep calibration drift", lambda: exp_sweep.main([*od, *mf]))
    step("3/10 nguong phat hien (alignment score)", lambda: exp_detect.main(od))
    step("4/10 failure case", lambda: exp_failure.main([*od, "--max-nusc-frames", str(args.max_nusc_frames)]))
    if not args.skip_latency:
        step("5/10 latency p50/p95", lambda: exp_latency.main(od))
    if not args.skip_bonus:
        step("6/10 B1: so sanh 2 thuat toan alignment score", lambda: exp_compare.main([*od, *mf]))
        step("7/10 B2: stress test suy giam du lieu", lambda: exp_stress.main([*od, *mf]))
        step("8/10 B6: loi cai san trong data/synthetic", lambda: exp_synthetic.main(od))
    if not args.skip_determinism:
        step("9/10 kiem tra tai lap", lambda: determinism_check(out))
    if not args.skip_report:
        step("10/10 sinh REPORT.md", lambda: make_report.main(
            ["--results-dir", str(out), "--out", args.report_out, "--class-name", args.class_name,
             "--student-name", args.student_name]))
    print("\nXong. Chay `python tools/check_submission.py` de kiem tra hinh thuc.")


if __name__ == "__main__":
    main()
