import os
import sys
from scripts.runpod_runner import run_on_runpod

def main():
    job_dir = os.path.abspath("simulations/test_runpod")
    input_file = "in.test"
    pot_path = os.path.abspath("potentials/Cu_zhou.eam.alloy")
    
    if not os.path.exists(pot_path):
        print(f"Error: Potential file not found at {pot_path}")
        return

    print("Starting RunPod LAMMPS Connection Test...")
    
    try:
        run_on_runpod(
            job_dir=job_dir,
            input_file=input_file,
            pot_path=pot_path,
            np=1,
            keep_pod=False
        )
        print("\nTest sequence finished.")
    except Exception as e:
        print(f"\nTest failed with error: {e}")

if __name__ == "__main__":
    main()
