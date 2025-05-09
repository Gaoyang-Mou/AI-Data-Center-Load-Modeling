import pandas as pd
from google.cloud import bigquery
import time
import logging
import sys
import gc # Garbage collector
import random # For sampling machines

# --- Configuration ---
GCP_PROJECT_ID = 'sonorous-cacao-459217-k1'
PUBLIC_DATA_PROJECT_ID = "google.com:google-cluster-data"
TARGET_CELL = 'a'
WORKLOAD_DATASET_ID = f"clusterdata_2019_{TARGET_CELL}"
POWER_DATASET_ID = "powerdata_2019"
WORKLOAD_TABLE_ID = "instance_usage"
MAPPING_TABLE_ID = "machine_to_pdu_mapping"

# *** MODIFIED CONFIGURATION ***
DURATION_IN_DAYS = 30 # Aim for a month (approx. 30 days)
# Select PDUs from which to draw your machine sample
# This helps keep the power analysis somewhat focused.
# If these PDUs have > TARGET_MACHINE_COUNT machines, we will sample.
# If they have <= TARGET_MACHINE_COUNT, we use all machines from these PDUs.
SELECTED_PDUS_FOR_MACHINE_POOL = ['pdu6', 'pdu7'] # Example: PDUs in cell 'a'
TARGET_MACHINE_COUNT = 100 # Target number of machines to analyze

# Output filename will reflect the strategy
pdu_pool_str = "_".join(sorted(SELECTED_PDUS_FOR_MACHINE_POOL))
OUTPUT_FILENAME = f"merged_data_cell_{TARGET_CELL}_{pdu_pool_str}_approx{TARGET_MACHINE_COUNT}machines_{DURATION_IN_DAYS}d.feather"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

# --- Helper Functions (query_bigquery and process_timestamps remain the same) ---
def query_bigquery(client, query_string, job_description):
    logging.info(f"Running query: {job_description}")
    start_time = time.time()
    try:
        if client is None:
             client = bigquery.Client(project=GCP_PROJECT_ID)
        query_job = client.query(query_string)
        df = query_job.to_dataframe(progress_bar_type='tqdm')
        end_time = time.time()
        logging.info(f"Query successful for {job_description}. Shape: {df.shape}. Time: {end_time - start_time:.2f}s")
        logging.info(f"DataFrame Memory Usage: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
        return df
    except Exception as e:
        logging.error(f"Error during query '{job_description}': {e}", exc_info=True)
        return pd.DataFrame()

def process_timestamps(df, time_col_name):
    if time_col_name in df.columns:
        logging.info(f"Processing timestamp column: {time_col_name}")
        try:
            df['datetime'] = pd.to_datetime(df[time_col_name], unit='us')
            logging.info(f"Successfully added 'datetime' column from '{time_col_name}'.")
        except Exception as e:
             logging.error(f"Error converting timestamp column '{time_col_name}': {e}", exc_info=True)
    else:
        logging.warning(f"Timestamp column '{time_col_name}' not found in DataFrame.")
    return df

# --- Main Data Processing ---
def main():
    logging.info(f"--- Starting Data Processing Pipeline (Long Duration, ~{TARGET_MACHINE_COUNT} Sampled Machines) ---")
    overall_start_time = time.time()

    client = None
    try:
        client = bigquery.Client(project=GCP_PROJECT_ID)
        logging.info(f"BigQuery client initialized for project: {GCP_PROJECT_ID}")
    except Exception as e:
        logging.error(f"Failed to initialize BigQuery client: {e}", exc_info=True)
        return

    # --- Define Time Range ---
    day_in_micros = 24 * 3600 * 1000000
    start_timestamp_micros = 600 * 1000000
    end_timestamp_micros = start_timestamp_micros + (DURATION_IN_DAYS * day_in_micros)
    logging.info(f"Target time range: {pd.to_datetime(start_timestamp_micros, unit='us')} to {pd.to_datetime(end_timestamp_micros, unit='us')} ({DURATION_IN_DAYS} days)")

    # --- 1. Query Machine to PDU Mapping ---
    query_string_mapping = f"""
        SELECT CAST(machine_id AS INT64) AS machine_id, pdu, cell
        FROM `{PUBLIC_DATA_PROJECT_ID}.{POWER_DATASET_ID}.{MAPPING_TABLE_ID}`
        WHERE cell = '{TARGET_CELL}'
    """
    df_mapping_full = query_bigquery(client, query_string_mapping, "Machine->PDU Mapping")
    if df_mapping_full.empty:
        logging.error("Failed to fetch mapping data. Exiting.")
        return

    # --- 2. Select Machine Subset ---
    logging.info(f"Selecting machines from PDUs: {SELECTED_PDUS_FOR_MACHINE_POOL}")
    machines_in_selected_pdus_df = df_mapping_full[df_mapping_full['pdu'].isin(SELECTED_PDUS_FOR_MACHINE_POOL)]
    
    if machines_in_selected_pdus_df.empty:
        logging.error(f"No machines found in the selected PDUs: {SELECTED_PDUS_FOR_MACHINE_POOL}. Exiting.")
        return
        
    all_machine_ids_from_pool = machines_in_selected_pdus_df['machine_id'].unique().tolist()
    logging.info(f"Found {len(all_machine_ids_from_pool)} unique machines in PDUs {SELECTED_PDUS_FOR_MACHINE_POOL}.")

    if len(all_machine_ids_from_pool) > TARGET_MACHINE_COUNT:
        logging.info(f"Sampling {TARGET_MACHINE_COUNT} machines from the {len(all_machine_ids_from_pool)} available machines.")
        final_selected_machine_ids = random.sample(all_machine_ids_from_pool, TARGET_MACHINE_COUNT)
    else:
        logging.info(f"Using all {len(all_machine_ids_from_pool)} machines found in selected PDUs (as it's <= target_machine_count).")
        final_selected_machine_ids = all_machine_ids_from_pool
    
    if not final_selected_machine_ids:
        logging.error("No machines selected for analysis. Exiting.")
        return
    logging.info(f"Final number of machines selected for analysis: {len(final_selected_machine_ids)}")
    machine_id_list_string = ", ".join(map(str, final_selected_machine_ids))

    # Determine the actual PDUs these selected machines belong to (should be SELECTED_PDUS_FOR_MACHINE_POOL)
    # This is important for querying the correct power data.
    actual_pdus_for_selected_machines = df_mapping_full[df_mapping_full['machine_id'].isin(final_selected_machine_ids)]['pdu'].unique().tolist()
    logging.info(f"The {len(final_selected_machine_ids)} selected machines belong to PDUs: {actual_pdus_for_selected_machines}")


    # --- 3. Query Workload Data for Selected Machine Subset ---
    query_string_task_usage = f"""
        SELECT
            start_time,
            CAST(machine_id AS INT64) AS machine_id,
            average_usage.cpus AS avg_cpu_usage,
            average_usage.memory AS avg_memory
        FROM
            `{PUBLIC_DATA_PROJECT_ID}.{WORKLOAD_DATASET_ID}.{WORKLOAD_TABLE_ID}`
        WHERE
            start_time >= {start_timestamp_micros} AND start_time < {end_timestamp_micros}
            AND machine_id IS NOT NULL
            AND CAST(machine_id AS INT64) IN ({machine_id_list_string})
    """
    df_task_usage_subset = query_bigquery(client, query_string_task_usage, f"Task Usage ({DURATION_IN_DAYS}d, {len(final_selected_machine_ids)} machines)")
    if df_task_usage_subset.empty:
        logging.error("Failed to fetch task usage data for the selected machine subset/time range. Exiting.")
        return

    # --- 4. Query Power Data (Only for PDUs housing the selected machines) ---
    all_power_df_list = []
    # Use 'actual_pdus_for_selected_machines' to ensure we only query relevant power tables
    power_pdu_tables_to_query = [f"cell{TARGET_CELL}_{pdu_id}" for pdu_id in actual_pdus_for_selected_machines]
    logging.info(f"Planning to query power tables: {power_pdu_tables_to_query}")

    for power_table in power_pdu_tables_to_query:
        query_string_power = f"""
            SELECT time, cell, pdu, measured_power_util, production_power_util
            FROM `{PUBLIC_DATA_PROJECT_ID}.{POWER_DATASET_ID}.{power_table}`
            WHERE time >= {start_timestamp_micros} AND time < {end_timestamp_micros}
        """
        df_power_single_pdu = query_bigquery(client, query_string_power, f"Power Data ({power_table})")
        if not df_power_single_pdu.empty:
            all_power_df_list.append(df_power_single_pdu)
        else:
             logging.warning(f"No power data returned for table {power_table} in the specified time range.")

    if not all_power_df_list:
         logging.warning("No power data fetched for the PDUs of selected machines. Merged data will lack power info.")
         df_power_subset = pd.DataFrame() # Create empty if no power data
    else:
        df_power_subset = pd.concat(all_power_df_list, ignore_index=True)
        logging.info(f"Combined power data for relevant PDUs. Total shape: {df_power_subset.shape}")
    del all_power_df_list
    gc.collect()

    # --- 5. Process Timestamps ---
    df_task_usage_subset = process_timestamps(df_task_usage_subset, 'start_time')
    if not df_power_subset.empty: # Only process if power data was fetched
        df_power_subset = process_timestamps(df_power_subset, 'time')

    # --- 6. Aggregate Workload per Machine (15 min) ---
    workload_agg_per_machine_15min = pd.DataFrame()
    if not df_task_usage_subset.empty and 'datetime' in df_task_usage_subset.columns:
        logging.info("Aggregating workload per machine to 15min (SUBSET)...")
        try:
            df_task_usage_subset['machine_id'] = pd.to_numeric(df_task_usage_subset['machine_id'], errors='coerce').astype('int64')
            workload_agg_per_machine_15min = df_task_usage_subset.set_index('datetime').groupby(['machine_id']).resample('15min').agg(
                avg_cpu_usage = ('avg_cpu_usage', 'mean'),
                avg_memory = ('avg_memory', 'mean')
            ).reset_index()
            logging.info(f"Workload aggregation per machine (SUBSET) complete. Shape: {workload_agg_per_machine_15min.shape}")
        except Exception as e:
            logging.error(f"Error during subset workload aggregation per machine: {e}", exc_info=True)
    del df_task_usage_subset; gc.collect()

    # --- 7. Aggregate Power per PDU (15 min) ---
    df_power_agg_per_pdu_15min = pd.DataFrame()
    if not df_power_subset.empty and 'datetime' in df_power_subset.columns:
        logging.info("Aggregating power per PDU to 15min (SUBSET)...")
        try:
            df_power_agg_per_pdu_15min = df_power_subset.set_index('datetime').groupby(['pdu', 'cell']).resample('15min').agg(
                measured_power_util = ('measured_power_util', 'mean'),
                production_power_util = ('production_power_util', 'mean')
            ).reset_index()
            logging.info(f"Power aggregation per PDU (SUBSET) complete. Shape: {df_power_agg_per_pdu_15min.shape}")
        except Exception as e:
            logging.error(f"Error during subset power aggregation: {e}", exc_info=True)
    del df_power_subset; gc.collect()

    # --- 8. Merge Data ---
    df_merged_final = pd.DataFrame()
    if not workload_agg_per_machine_15min.empty and not df_mapping_full.empty: # df_mapping_full used here for full mapping context
        logging.info("Merging aggregated workload and power data (SUBSET)...")
        try:
            # Use the full mapping to get PDU for each selected machine
            df_mapping_full['machine_id'] = pd.to_numeric(df_mapping_full['machine_id'], errors='coerce').astype('int64')
            workload_with_pdu = pd.merge(workload_agg_per_machine_15min,
                                         df_mapping_full[df_mapping_full['machine_id'].isin(final_selected_machine_ids)], # Filter mapping to only selected machines
                                         on='machine_id', how='left')
            workload_with_pdu.dropna(subset=['pdu','cell'], inplace=True)
            logging.info(f"Shape after merge workload + mapping (SUBSET): {workload_with_pdu.shape}")
            del workload_agg_per_machine_15min; gc.collect()

            logging.info("Aggregating workload per PDU (SUBSET)...")
            workload_agg_per_pdu = workload_with_pdu.groupby(['datetime', 'pdu', 'cell']).agg(
                pdu_avg_cpu_usage = pd.NamedAgg(column='avg_cpu_usage', aggfunc='mean'),
                pdu_sum_cpu_usage = pd.NamedAgg(column='avg_cpu_usage', aggfunc='sum'),
                pdu_avg_memory_usage = pd.NamedAgg(column='avg_memory', aggfunc='mean'),
                pdu_sum_memory_usage = pd.NamedAgg(column='avg_memory', aggfunc='sum'),
                machine_count_in_pdu_subset = pd.NamedAgg(column='machine_id', aggfunc='nunique') # No. of *selected* machines in this PDU
            ).reset_index()
            logging.info(f"Workload aggregation per PDU (SUBSET) complete. Shape: {workload_agg_per_pdu.shape}")
            del workload_with_pdu; gc.collect()

            if not df_power_agg_per_pdu_15min.empty:
                logging.info("Merging PDU workload with PDU power (SUBSET)...")
                df_merged_final = pd.merge(
                    workload_agg_per_pdu,
                    df_power_agg_per_pdu_15min,
                    on=['datetime', 'pdu', 'cell'],
                    how='left' # Left join to keep all workload data, power is supplemental
                )
                logging.info(f"Final merge (SUBSET) complete. Final DataFrame shape: {df_merged_final.shape}")
            else:
                logging.warning("No power data to merge. Final data will only contain workload.")
                df_merged_final = workload_agg_per_pdu # Use workload data if no power data
        except Exception as e:
            logging.error(f"Error during merging (SUBSET): {e}", exc_info=True)
    else:
        logging.warning("Skipping final merge (SUBSET) as precursor DataFrames are empty.")

    # --- 9. Save Final Result ---
    if not df_merged_final.empty:
        logging.info(f"Saving final merged subset data to {OUTPUT_FILENAME}...")
        try:
            df_merged_final.to_feather(OUTPUT_FILENAME)
            logging.info(f"Save successful to {OUTPUT_FILENAME}.")
        except Exception as e:
            logging.error(f"Error saving data to {OUTPUT_FILENAME}: {e}", exc_info=True)
    else:
        logging.warning("Final merged DataFrame (SUBSET) is empty, nothing to save.")

    overall_end_time = time.time()
    logging.info(f"--- Data Processing Pipeline (Long Duration, Sampled Machines) Finished in {overall_end_time - overall_start_time:.2f} seconds ---")

if __name__ == "__main__":
    main()