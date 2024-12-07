import wandb
import pandas as pd
from math import ceil

def fetch_wandb_data(run_paths, metric='next_batch_acc1_step'):
    """
    Fetches specified metric data from multiple WandB runs and formats it into a DataFrame.

    Parameters:
    - run_paths (list of str): List of WandB run paths in the format "entity/project/run_id".
    - metric (str): The metric to fetch from the runs.

    Returns:
    - pandas.DataFrame: A DataFrame with time steps as rows and each run as a column.
    """
    # Initialize a dictionary to store metric data for each run
    data = {}

    # Iterate over each run path
    for run_path in run_paths:
        # Split the run path to extract entity, project, and run ID
        entity, project, run_id = run_path.split('/')

        # Initialize WandB API
        api = wandb.Api()

        # Fetch the run object
        run = api.run(run_path)

        # Extract the metric data
        # keep only 10 rows
        history = run.history(keys=[metric, "timestep_step"], samples=500)

        # Retrieve metric values and time steps
        metric_values = history[metric].tolist()
        time_steps = history["timestep_step"].tolist()

        # Select uniformly spaced 10 rows
        step_size = len(time_steps) // 10
        metric_values = metric_values[::step_size]
        time_steps = time_steps[::step_size]

        # round to the nearest 5000
        time_steps = [int(5000 * ceil(step/5000)) for step in time_steps]

        # Store data in the dictionary using the run name as the key
        run_name = run.name if run.name else run_id
        data[run_name] = metric_values

    # Create a DataFrame with time steps as the index
    df = pd.DataFrame(data)
    df.insert(0, 'Time Steps', time_steps)

    return df

def print_markdown_table(df):
    """
    Prints the DataFrame in Markdown table format.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the metric data.
    """
    # Create a Markdown table header
    header = "| " + " | ".join(df.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"

    # Initialize the list to store each row of the table
    rows = [header, separator]

    # Format the DataFrame values
    df_formatted = df.copy()
    df_formatted.iloc[:, 1:] = df_formatted.iloc[:, 1:].applymap(lambda x: f"{x:.3f}")
    df_formatted['Time Steps'] = df_formatted['Time Steps'].apply(lambda x: f"{int(x)}")

    # Add each row of the formatted DataFrame to the list
    for index, row in df_formatted.iterrows():
        rows.append("| " + " | ".join(map(str, row)) + " |")

    # Join all the rows with newline characters to form the Markdown table
    markdown_table = "\n".join(rows)

    # Print the Markdown table
    print("Copy the following table to OpenReview:\n")
    print(markdown_table)


# Example usage
run_paths = [
    "[author1]/onlineCL-cs1/1gin2rzn",
    '[author1]/onlineCL-cs1/72bjdska',
]

# Fetch the data
df = fetch_wandb_data(run_paths)

# Print the table
print_markdown_table(df)