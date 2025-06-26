# data_pipeline/schedule_realtime.py
import subprocess
import time

if __name__ == "__main__":
    while True:
        # เรียก trade_executor ในแบบ realtime
        p = subprocess.Popen([
            "python", "-u", "-m", "data_pipeline.trade_executor",
            "--config", "config.yaml"
        ])
        ret = p.wait()
        if ret != 0:
            print(f"Executor จบด้วยโค้ด {ret} รอ 10 วินาที แล้วรันใหม่...")
            time.sleep(10)
        else:
            break
