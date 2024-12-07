import wandb
import sys
import subprocess
import time
import filelock

sweep_id = sys.argv[1]
with filelock.FileLock(".lock"):
    api = wandb.Api()
    runs = api.sweep(sweep_id).runs

    # filter crashed runs
    crashed_runs = [run for run in runs if run.state in ["crashed", "failed"]]

    # make sure that the crashed runs are not currently being relaunched (in case of batch jobs)
    crashed_runs = [run for run in crashed_runs if run.summary.get("resume_state") != "configuring"]

    if len(crashed_runs) == 0:
        print("No crashed runs found.")
        exit(0)

    # sort by most recent
    crashed_runs = sorted(crashed_runs, key=lambda run: run.lastHistoryStep, reverse=True)

    print("Crashed runs:")
    for run in crashed_runs:
        print(run.id)

    run = crashed_runs[0]
    print("Resuming run:", run.id)
    run.summary["resume_state"] = "configuring"
    run.update()


    print("Waiting for run to be configured...")
    for i in range(100):
        new_runs = api.sweep(sweep_id).runs
        new_crashed_runs = [new_run for new_run in new_runs if new_run.state in ["crashed", "failed"]]
        new_crashed_runs = [new_run for new_run in new_crashed_runs if new_run.summary.get("resume_state") != "configuring"]
        if run.id in new_crashed_runs:
            print(f"Run is still configuring. Waiting for wandb to sync... {i}/100")
            time.sleep(10)
        else:
            break


print("Resuming run:", run.id)
run.summary["resume_state"] = "resumed"
run.update()


# create command from config
command = f"python main_delay.py +resume_wandb_run={run.id}"

print("Running command:")
print(command)


# run command with subprocess
try:
    run.summary["resume_state"] = "resumed"
    run.update()
    subprocess.run(command.split(" "))
except Exception as e:
    print(e)
    run.summary["resume_state"] = "crashed"
    run.update()
    exit(1)

run.summary["resume_state"] = "finished"
run.update()

