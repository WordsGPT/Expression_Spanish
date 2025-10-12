# Expression Spanish

## Initial configuration

1. Install dependencies by running:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure your OpenAI API key by copying `apis_example.env` to `apis.env` and setting your API key:
   ```bash
   cp apis_example.env apis.env
   ```

   Then edit `apis.env` and replace `"your-api-key"` with your actual OpenAI API key.

## Experiment structure

Each experiment folder should contain:

- `config.yaml`: Configuration file with experiment definitions
- `data/`: Dataset files (CSV or Excel)
- `prompts/`: Text files containing prompts for each experiment
- `batches/`: Generated batch files (created by prepare_experiment.py)
- `results/`: Results from OpenAI API (downloaded by execute_experiment.py)
- `outputs/`: Final processed results (created by generateResults.py)

## Make estimations with some model:

1. Prepare the experiment by running:
   ```bash
   python prepare_experiment.py <EXPERIMENT_NAME> <EXPERIMENT_PATH>
   ```
   This generates the batch files from your configuration.

   **Available EXPERIMENT_NAME options:**
   | Option | Description |
   |--------|-------------|
   | `<experiment_name>` | Process specific experiment |
   | `"all"` | Process all experiments |
   | `"failed"` | Retry failed experiments |
   | `"status"` | Show failed experiments |

2. Run the experiment by executing:
   ```bash
   python execute_experiment.py <EXPERIMENT_NAME> <EXPERIMENT_PATH>
   ```
   This executes the batches and automatically downloads results when completed.

   **Available EXPERIMENT_NAME options:**
   | Option | Description |
   |--------|-------------|
   | `<experiment_name>` | Process specific experiment |
   | `"all"` | Process all experiments in the batches folder |
   | `"failed"` | Retry failed experiments |
   | `"remain"` | Check and download batches still in tracking |
   | `"status"` | Show batches still in tracking |

## Generating results:

1. Execute:
   ```bash
   python generateResults.py <EXPERIMENT_NAME> <EXPERIMENT_PATH>
   ```
   This processes the downloaded results and generates a .xlsx file in the folder `outputs/<EXPERIMENT NAME>`

   **Available EXPERIMENT_NAME options:**
   | Option | Description |
   |--------|-------------|
   | `<experiment_name>` | Process specific experiment |
   | `"all"` | Process all experiments |
   | `"failed"` | Retry failed experiments |
   | `"status"` | Show failed experiments |
