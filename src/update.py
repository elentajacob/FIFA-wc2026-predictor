"""
Live updater for the WC2026 predictor.

"""

import subprocess
import sys
from datetime import datetime


def run_step(label, script):
    print(f"\n{'='*60}")
    print(f"STEP: {label}")
    print('='*60)
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"ERROR in {script}:")
        print(result.stderr)
        return False
    return True


def main():
    print(f"\nWC2026 Predictor — Live Update")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if not run_step("Fetching latest WC2026 fixtures and results", "src/fetch_data.py"):
        return
    if not run_step("Refreshing predictions for remaining matches", "src/predict.py"):
        return

    print(f"\n{'='*60}")
    print("UPDATE COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('='*60)
    print("\nRestart Streamlit to see the refreshed dashboard:")
    print("  streamlit run app.py")


if __name__ == "__main__":
    main()