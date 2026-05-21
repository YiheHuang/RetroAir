"""
run_pipeline.py —— 一键运行 MVP 完整管线
用法:
  python run_pipeline.py --api-key YOUR_OPENAQ_KEY
  python run_pipeline.py --api-key YOUR_OPENAQ_KEY --test-mode
  python run_pipeline.py --api-key YOUR_OPENAQ_KEY --skip-fetch
"""
import sys
import subprocess
import time
import os
from pathlib import Path

ROOT = Path(__file__).parent
SCRIPTS = ROOT / "scripts"


def run(script: str, args: list = None):
    cmd = [sys.executable, str(SCRIPTS / script)]
    if args:
        cmd.extend(args)
    name = script.replace(".py", "").replace("_", " ")
    print(f"\n{'='*60}")
    print(f">>> {name}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT), env={**os.environ})
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n  FAILED (exit {result.returncode})")
        return False
    print(f"\n  OK ({elapsed:.0f}s)")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="UrbanAir MVP 一键管线")
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--skip-osm", action="store_true")
    parser.add_argument("--skip-app", action="store_true")
    parser.add_argument("--max-stations", type=int, default=50)
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════╗")
    print("║       UrbanAir MVP Pipeline                         ║")
    print("║       城市环境质量多模态评估系统                      ║")
    print("╚══════════════════════════════════════════════════════╝")

    # 构建各步骤参数
    if args.test_mode:
        date_args = ["--start", "2024-06-01", "--end", "2024-06-30"]
        weather_extra = []
    else:
        date_args = ["--start", "2024-01-01", "--end", "2024-12-31"]
        aqi_extra = ["--max-stations", str(args.max_stations)]
        weather_extra = []

    steps = []

    # Step 1-2: 数据下载
    if not args.skip_fetch:
        steps.append(("AQI 数据", "01_fetch_aqi.py", date_args))
        steps.append(("气象数据", "02_fetch_weather.py", date_args + weather_extra))

    # Step 3: OSM 特征
    if not args.skip_osm:
        steps.append(("OSM 特征提取", "03_extract_osm.py", []))

    # Step 4-5: 合并 + 训练
    steps += [
        ("数据合并", "04_merge_data.py", []),
        ("模型训练", "05_train_model.py", []),
    ]

    # 执行
    failed = False
    for i, (desc, script, extra) in enumerate(steps):
        print(f"\n[{i+1}/{len(steps)}] {desc}")
        if not run(script, extra):
            failed = True
            break

    if failed:
        print("\n管线中断, 请检查上方错误信息.")
        sys.exit(1)

    # 输出文件清单
    print(f"\n{'='*60}")
    print("管线完成! 输出文件:")
    print(f"{'='*60}")
    data_dir = ROOT / "data"
    for p in sorted(data_dir.rglob("*")):
        if p.is_file() and not p.name.startswith(".") and ".cache" not in str(p):
            size_kb = p.stat().st_size / 1024
            rel = p.relative_to(ROOT)
            print(f"  {str(rel):<50s} {size_kb:>8.1f} KB")

    # 启动 App
    if not args.skip_app:
        print(f"\n{'='*60}")
        print("启动 Streamlit 应用...")
        print(f"  cd {ROOT}")
        print(f"  streamlit run scripts/06_app.py")
        print(f"{'='*60}")
        subprocess.run(
            ["streamlit", "run", str(SCRIPTS / "06_app.py")],
            cwd=str(ROOT),
            env={**os.environ},
        )


if __name__ == "__main__":
    main()
