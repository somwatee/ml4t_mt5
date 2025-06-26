# ── data_pipeline/schedule_realtime.py ─────────────────────────────
import subprocess, time

while True:
    p = subprocess.Popen(["python","-u","trade_executor.py","config.yaml"])
    ret = p.wait()
    if ret != 0:
        print(f"Executor exited ({ret}), restarting in 10s...")
        time.sleep(10)
    else:
        break
