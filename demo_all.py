import subprocess
import sys
import os
import time

# Get the directory where THIS script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def print_separator(title):
    width = 60
    print("\n" + "#" * width)
    print(f"#{title.center(width - 2)}#")
    print("#" * width + "\n")


def run_command(command, cwd=None, extra_pythonpath=None):
    """
    Runs a command safely.
    Catches KeyboardInterrupt (Ctrl+C) to allow skipping ONLY the current demo.
    """
    try:
        env = os.environ.copy()

        # Construct PYTHONPATH
        current_path = env.get("PYTHONPATH", "")
        parts = [current_path, BASE_DIR]
        if extra_pythonpath:
            parts.append(extra_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, parts))

        print(f"   >>> Running... (Press Ctrl+C to SKIP this demo)")

        # We start the process
        subprocess.run(command, env=env, cwd=cwd)

        # Small pause between successful runs
        time.sleep(1)

    except KeyboardInterrupt:
        # This block catches the Ctrl+C
        print("\n\n⚠️  SKIPPING CURRENT DEMO! Moving to the next one...")

        # --- CRITICAL FIX ---
        # We wait a moment to let the user release the keys.
        # We wrap this sleep in a try/except so holding Ctrl+C doesn't crash the script here.
        try:
            time.sleep(1.5)
        except KeyboardInterrupt:
            pass  # Ignore extra Ctrl+C presses during the transition

    except Exception as e:
        print(f"❌ Error running command: {e}")


def find_file(filename, search_subfolders):
    # Check root first
    if os.path.exists(os.path.join(BASE_DIR, filename)):
        return os.path.join(BASE_DIR, filename)
    # Check subfolders
    for folder in search_subfolders:
        path = os.path.join(BASE_DIR, folder, filename)
        if os.path.exists(path):
            return path
    return None


def main():
    print_separator("STARTING ALL RACETRACK DEMOS")
    print("TIP: You can press Ctrl+C anytime to skip to the next model.\n")

    # --- 1. ACTOR-CRITIC ---
    print_separator("1. ACTOR-CRITIC (PPO)")
    script = find_file("actor_critique.py", ["Actor Critique", "ActorCritique"])
    if script:
        run_command([sys.executable, script, "--mode", "demo"], cwd=os.path.dirname(script))
    else:
        print("❌ Script not found.")

    # --- 2. BEHAVIOR CLONING ---
    print_separator("2. BEHAVIOR CLONING")
    script = find_file("3_demo_final.py", ["Behavior Cloning", "BehaviorCloning"])
    if script:
        run_command([sys.executable, script], cwd=os.path.dirname(script))
    else:
        print("❌ Script not found.")

    # --- 3. DYNA-Q ---
    print_separator("3. DYNA-Q (Model-Based)")
    script = find_file("last_version_1.py", ["Dyna-Q", "DynaQ"])
    if script:
        script_dir = os.path.dirname(script)
        # Run using python -c to import only the demo function
        cmd = [sys.executable, "-c", "from last_version_1 import ruleaza_demo; ruleaza_demo()"]
        run_command(cmd, cwd=script_dir, extra_pythonpath=script_dir)
    else:
        print("❌ Script not found.")

    # --- 4. DEEP Q-NETWORK (DQN) ---
    print_separator("4. DEEP Q-NETWORK (DQN)")
    script = find_file("run_dqn.py", ["Q-Learn & DQN", "Q-Learn & DQN/dqn"])
    if script:
        run_command([sys.executable, script], cwd=os.path.dirname(script))
    else:
        print("❌ Script not found.")

    # --- 5. Q-LEARNING ---
    print_separator("5. Q-LEARNING (Table-Based)")
    script = find_file("run.py", ["Q-Learn & DQN", "Q-Learn & DQN/qlearning", "qlearning"])
    if script:
        run_command([sys.executable, script], cwd=os.path.dirname(script))
    else:
        print("❌ Script not found.")

    print_separator("ALL DEMOS COMPLETED")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n>>> Exiting All Demos.")