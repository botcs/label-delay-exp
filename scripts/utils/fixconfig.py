import wandb
wandb.login()
api = wandb.Api(timeout=100)
api_hash = hash(api)
print(f"API hash: {api_hash}")

def fix_missing_values(sweep_id, key, value):

    # Fetch the sweep object
    sweep = api.sweep(sweep_id)

    # Fetch the runs from the sweep
    runs = sweep.runs
    # Iterate over the runs
    fixed_runs = 0
    for run in runs:
        print(f"Checking run {run}")
        if run.config.get("method") in ["base"]:
            print(f"Fixing '{key}={value}' for run: {run}")
            run.config[key] = value
            run.config["api_tinker_hash"] = str(api_hash) + ";" + str(run.config.get("api_tinker_hash") or "")
            run.update()
            fixed_runs += 1

    # Print a message after fixing the missing values
    print(f"Missing values fixed for {fixed_runs} runs")

def fix_missing_values(sweep_id, key, value):

    # Fetch the sweep object
    sweep = api.sweep(sweep_id)

    # Fetch the runs from the sweep
    runs = sweep.runs
    # Iterate over the runs
    fixed_runs = 0
    for run in runs:
        print(f"Checking run {run}")
        if not run.config.get("method") in ["base", "tent"]:
            print(f"Fixing '{key}={value}' for run: {run}")
            run.config[key] = value
            run.config["api_tinker_hash"] = str(api_hash) + ";" + str(run.config.get("api_tinker_hash") or "")
            run.update()
            fixed_runs += 1

    # Print a message after fixing the missing values
    print(f"Missing values fixed for {fixed_runs} runs")

def set_crashed(sweep_id):

    # Fetch the sweep object
    sweep = api.sweep(sweep_id)

    # Fetch the runs from the sweep
    runs = sweep.runs
    # Iterate over the runs
    fixed_runs = 0
    for run in runs:
        if run.summary.get("next_batch_acc1_epoch") is None:
            run.state = "crashed"
            run.update()
            fixed_runs += 1

    print(fixed_runs)


def update_summary_to_latest_history(sweep_id, keys=["trainer/global_step", "timestep_step"], dry_run=True):

    # Fetch the sweep object
    sweep = api.sweep(sweep_id)

    # Fetch the runs from the sweep
    runs = sweep.runs
    # Iterate over the runs
    for run in runs:
        print(f"Checking run: {run}")
        # latest_history = run.history(keys=keys, pandas=False)[-1]
        last_step = run.lastHistoryStep
        print(f"Last step: {last_step}")
        latest_history = list(run.scan_history(keys=keys+["_step"], min_step=last_step-1000, page_size=1000))
        
        # sort by _step
        latest_history = sorted(latest_history, key=lambda x: x["_step"])

        # for each key find the last value
        last_values = {}
        for history in latest_history:
            for key in keys:
                if key in history:
                    last_values[key] = history[key]

        
        print(f"Latest history: {last_values}")
        for key in keys:
            if key in last_values:
                if run.summary.get(key) != last_values[key]:
                    
                    print(f"Updating {key:20}: {str(run.summary[key]):10} -> {str(last_values[key]):10}")
                    run.summary[key] = last_values[key]
        if not dry_run:
            run.update()

        
