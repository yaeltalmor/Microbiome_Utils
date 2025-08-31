from xml.sax import make_parser

import numpy as np
import pandas as pd
from sheeba_utils.config import *
#
# ############ Category-Based Format Data Utils ############
# train_window_len - number of months for training (24, 36, 48, 60)
# last_month_index = train_window_len - 1 (the dates are zero based)
def masking_rule(df: pd.DataFrame, col_to_mask: str, threshold: int, intervals_from_diagnosis: bool):
    assert 'end_of_data' in df.columns
    if intervals_from_diagnosis:
        masked_col = df[col_to_mask].where(df[col_to_mask] < threshold, np.nan)
    else:
        masked_col = df[col_to_mask].where(df[col_to_mask] > df.end_of_data - threshold, np.nan)
    return masked_col


def compute_events_aggregations(cat_based_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes drug switches, duration average, procedures, and trajectory length for each patient.
    """
    scores = []

    for _, row in cat_based_df.iterrows():
        patient_id = row.name

        # Count Drug Changes (Unique switches)
        bio_drug_sequence = [row[drug] for drug in BIOLOGIC_DRUG_COLUMNS if pd.notna(row[drug])]
        num_biological_drugs = len(set(bio_drug_sequence))
        small_drug_sequence = [row[drug] for drug in SMALL_MOLECULE_DRUG_COLUMNS if pd.notna(row[drug])]
        num_small_molecule_drugs = len(set(small_drug_sequence))
        num_of_drugs = len(set(bio_drug_sequence)) + len(set(small_drug_sequence))

        # Drug Duration Variability (mean of drug durations)
        durations = [row[d] for d in DURATION_COLUMNS if pd.notna(row[d])]
        drug_duration_mean = np.mean(durations) if len(durations) > 1 else 0

        # Count Procedures
        procedure_count = sum(pd.notna(row[p]) for p in PROCEDURE_COLUMNS)
        # Count IBD related manifests
        ibd_related_mnfsts_count = sum(pd.notna(row[p]) for p in IBD_RELATED_MNFSTS)
        # Count non IBD conditions
        non_ibd_mnfsts_count = sum(pd.notna(row[p]) for p in NON_IBD_MNFSTS)

        # Count medical measures
        med_tests_count = sum(pd.notna(row[m]) for m in MED_TESTS_COLUMNS)

        gastro_visits_count = sum(pd.notna(row[g]) for g in GASTRO_VISITS_COLUMNS)

        # Compute Trajectory Length (max data exists in data aligned each patient month of diagnosis, +1 since it's zero based)
        trajectory_length = row.end_of_data + 1

        # Check if the patient has an active drug at the cutoff - uncomment if counters model run
        # active_drug_at_cutoff = row.get('active_drug_at_cutoff', False)

        scores.append({
            "patient_id": patient_id,
            "biological_drugs_count": num_biological_drugs,
            "small_molecule_drugs_count": num_small_molecule_drugs,
            "total_drugs_count": num_of_drugs,
            "drug_duration_mean": drug_duration_mean,
            "procedure_count": procedure_count,
            "ibd_related_mnfsts_count": ibd_related_mnfsts_count,
            "non_ibd_mnfsts_count": non_ibd_mnfsts_count,
            "med_tests_count": med_tests_count,
            "gastro_visits_count": gastro_visits_count,
            "trajectory_length": trajectory_length,
            # "active_drug_at_cutoff": active_drug_at_cutoff,
        })

    df_scores = pd.DataFrame(scores)
    df_scores['trajectory_length'] = df_scores['trajectory_length'].astype('Int64')  # Convert to Int64 with NaN support
    return df_scores


def calc_masked_events_agg(cat_based_df:pd.DataFrame, train_window_len:int, trajectory_length_threshold:int) -> pd.DataFrame:
    '''

    :param train_window_len: mask all patients events succeeding train_window_len
    :param trajectory_length_threshold: Filter patients with trajectory_length >= trajectory_length_threshold
    :return: df_events_agg: the events counters for the relevant patients history
    '''
    # Mask features by decile_month
    feature_cols = BIOLOGIC_DRUG_COLUMNS + SMALL_MOLECULE_DRUG_COLUMNS + PROCEDURE_COLUMNS \
                   + IBD_RELATED_MNFSTS + NON_IBD_MNFSTS + MED_TESTS_COLUMNS + GASTRO_VISITS_COLUMNS
    df_subset = cat_based_df[feature_cols + TRAJECTORY_COLUMNS].copy()
    for col in feature_cols:
        df_subset[col] = masking_rule(df_subset, col, train_window_len, intervals_from_diagnosis=True)
    # Mask duration columns based on corresponding drug dates
    # - not necessary, will implicitly be masked by mask_drug_durations
    df_durations = cat_based_df[DURATION_COLUMNS].copy()
    for i, duration_col in enumerate(DURATION_COLUMNS):
        drug_date_col = BIOLOGIC_DRUG_COLUMNS[i] if (i < len(BIOLOGIC_DRUG_COLUMNS)) & (
                'drug' in BIOLOGIC_DRUG_COLUMNS[i]) else None
        if drug_date_col:
            df_durations[duration_col] = df_durations[duration_col].where(df_subset[drug_date_col].notna(), np.nan)
    df_subset = df_subset.join(df_durations)
    # Trim durations to the train_window_len
    mask_drug_durations(train_window_len, df_subset)

    # Compute instability scores and use them as features
    df_events_agg = compute_events_aggregations(df_subset)
    # Filter patients with trajectory_length >= threshold
    df_events_agg = df_events_agg[df_events_agg["trajectory_length"] >= trajectory_length_threshold]

    # Filter patients with zero medical intervention
    print(f"df_events_agg num patients: {df_events_agg.shape}")
    zero_interventions_ids = get_zero_intervention_patients_id(df_events_agg)
    df_events_agg = df_events_agg.set_index('patient_id')
    df_events_agg = df_events_agg.loc[~df_events_agg.index.isin(zero_interventions_ids), :]
    print(f"df_events_agg num patients: {df_events_agg.shape}")


    return df_events_agg


def transform_exams_to_avg_per_year(df: pd.DataFrame, train_window_len: int) -> pd.DataFrame:
    '''
    Tranforms all multiple medical exams (crp, sccai_summary, calprotectin) into averaged values per year.
    :param df: df to add the transformed features ('{med_exam}_avg_year_{year_num}')
    :param train_window_len: number of years to calculate averaged values for in months
    :return: df with transformed features
    '''
    # Generate CRP average columns per year
    num_years = (train_window_len + 11) // 12  # Ceiling division

    for med_exam, mapper in MED_EXAMS_MAPPER.items():
        for year in range(num_years):
            start_month = year * 12
            end_month = start_month + 11

            def avg_med_exam_in_range(row):
                values = [
                    row[med_exam_col]
                    for date_col, med_exam_col in mapper.items()
                    if pd.notna(row[date_col]) and start_month <= row[date_col] <= end_month
                ]
                return np.mean(values) if values else np.nan

            df[f'{med_exam}_avg_year_{year + 1}'] = df.apply(avg_med_exam_in_range, axis=1)
    return df

def create_latest_event_value_encoding(masked_cat_based_df):
    # Define relevant prefixes for each category
    category_prefixes = {
        'procedure': ['procedure_1_category', 'procedure_2_category', 'last_procedure_category'],
        'extraintestinal_manifestations': [
            'extraintestinal_manifestations_organs_2',
            'extraintestinal_manifestations_organs_3',
            'extraintestinal_manifestations_organs_last'
        ],
        'neoplasms': [
            'neoplasms_diagnosis_2_category',
            'neoplasms_diagnosis_3_category',
            'last_neoplasms_diagnosis_category'
        ],
        'drug': [f'drug{i}' for i in range(1, 9)],
    }

    for category, prefixes in category_prefixes.items():
        # print(f"\n[DEBUG] Processing category: {category}")

        # Find all matching columns and map them to (prefix, suffix)
        prefix_suffix_pairs = []
        for prefix in prefixes:
            for col in masked_cat_based_df.columns:
                if col.startswith(prefix + '_'):
                    suffix = col[len(prefix) + 1:]  # remove prefix + underscore
                    prefix_suffix_pairs.append((prefix, suffix))

        # cat_hot_cols = [
        #     f"{prefix}_{suffix}" for prefix, suffix in prefix_suffix_pairs
        # ]
        # print(f"[DEBUG]   Matched columns (cat_hot_cols): {cat_hot_cols}")

        # Collect unique suffixes
        cat_hot_cols_values = sorted(set(suffix for _, suffix in prefix_suffix_pairs))
        # print(f"[DEBUG]   Extracted value suffixes (cat_hot_cols_values): {cat_hot_cols_values}")

        for value in cat_hot_cols_values:
            if value in ['date', 'duration']:
                continue
            # Collect all columns from all prefixes that end with the current suffix
            single_value_cols = [
                f"{prefix}_{value}"
                for prefix in prefixes
                if f"{prefix}_{value}" in masked_cat_based_df.columns
            ]
            # print(f"[DEBUG]     Value: {value}")
            # print(f"[DEBUG]     Columns used for max (single_value_cols): {single_value_cols}")

            try:
                masked_cat_based_df[f'latest_{category}_{value}'] = masked_cat_based_df[single_value_cols].max(axis=1)
            except Exception as e:
                print(f"[ERROR] Failed to compute max for {category}_{value}: {e}")
                raise

    return masked_cat_based_df


def features_to_keep(masked_cat_based_df) -> list():
    # 1. Drug-related columns
    kept_drug_cols = SMALL_MOLECULE_DRUG_COLUMNS + [
                         col for col in masked_cat_based_df.columns
                         if col.startswith('drug') and col.endswith('_duration') or col.startswith('latest_drug_')
                     ] + ['active_drug_at_cutoff']

    # 2. Exam-related columns
    exam_prefixes = ['crp_avg_year_', 'sccai_summary_avg_year_', 'calprotectin_avg_year_']
    kept_exams_cols = [
        col for col in masked_cat_based_df.columns
        if col == 'bmi' or any(col.startswith(prefix) for prefix in exam_prefixes)
    ]

    # 3. Visits and colonoscopy columns
    kept_visits_cols = [
        col for col in masked_cat_based_df.columns
        if 'visit' in col.lower() or 'colonoscopy' in col.lower()
    ]

    # 4. Latest values from procedures, manifestations, neoplasms, and drug categories
    manifests_categories = ['procedure', 'extraintestinal_manifestations', 'neoplasms']
    kept_manifsts_prcds_cols = [
                                   col for col in masked_cat_based_df.columns
                                   if any(col.startswith(f'latest_{cat}_') for cat in manifests_categories)
                               ] + EX_MNFST_NON_IBD_COLS

    life_event_cols = ['birth_date', 'death_date']

    return life_event_cols, kept_drug_cols, kept_manifsts_prcds_cols, kept_exams_cols, kept_visits_cols


def drop_all_redandant_cols(masked_cat_based_df):
    life_event_cols, kept_drug_cols, kept_manifsts_prcds_cols, kept_exams_cols, kept_visits_cols = features_to_keep(
        masked_cat_based_df)
    masked_cat_based_df = masked_cat_based_df.loc[:,
                          life_event_cols + kept_drug_cols + kept_manifsts_prcds_cols + kept_exams_cols + kept_visits_cols]

    return masked_cat_based_df

def mask_drug_durations(train_window_len, masked_cat_based_df):
    active_drug_at_cutoff = pd.Series(False, index=masked_cat_based_df.index)
    very_early_month = -9999

    latest_drug_start_date = pd.Series(very_early_month, index=masked_cat_based_df.index)
    latest_drug_end_date = pd.Series(very_early_month, index=masked_cat_based_df.index)

    for i, duration_col in enumerate(DURATION_COLUMNS):
        drug_date_col = BIOLOGIC_DRUG_COLUMNS[i]
        drug_start = masked_cat_based_df[drug_date_col]

        # Default: until the end of the observation window (includes the month of prescription - len([12,23])=12)
        default_duration = train_window_len - drug_start

        # Duration until next drug, if it exists
        if i < len(DURATION_COLUMNS) - 1:
            next_drug_date_col = BIOLOGIC_DRUG_COLUMNS[i + 1]
            next_drug_start = masked_cat_based_df[next_drug_date_col]

            duration = (next_drug_start - drug_start).where(
                next_drug_start.notna(),
                default_duration
            )
        else:
            duration = default_duration

        # Save the masked duration
        masked_cat_based_df[duration_col] = duration

        # Compute drug end date
        drug_end = drug_start + duration - 1

        # Update latest drug info: only if current drug is more recent than any previous one
        is_newer = drug_start > latest_drug_start_date
        latest_drug_start_date = drug_start.where(is_newer, latest_drug_start_date)
        latest_drug_end_date = drug_end.where(is_newer, latest_drug_end_date)

    # Flag if the latest drug (per row) is still active at train_window_len
    last_month_idx = train_window_len - 1
    active_drug_at_cutoff = latest_drug_end_date == last_month_idx
    masked_cat_based_df['active_drug_at_cutoff'] = active_drug_at_cutoff.where(latest_drug_start_date.notna(), False)

    return

def filter_zero_medical_intervention_patients(masked_cat_based_df):
    df_counters = compute_events_aggregations(masked_cat_based_df)
    zero_interventions_ids = get_zero_intervention_patients_id(df_counters)
    masked_cat_based_df = masked_cat_based_df.loc[~masked_cat_based_df.index.isin(zero_interventions_ids), :]
    return masked_cat_based_df


def get_zero_intervention_patients_id(df_counters):
    zero_biologic_patients = set(df_counters[df_counters.biological_drugs_count == 0].patient_id)
    zero_small_drug_patients = set(df_counters[df_counters.small_molecule_drugs_count == 0].patient_id)
    zero_procedures_patients = set(df_counters[df_counters.procedure_count == 0].patient_id)
    zero_interventions_ids = list(
        (zero_biologic_patients).intersection(zero_small_drug_patients).intersection(zero_procedures_patients))
    return zero_interventions_ids


def mask_cat_based_data(cat_based_df:pd.DataFrame, train_window_len:int, trajectory_length_threshold:int) -> pd.DataFrame:
    '''
    This function masks the events according to the train_window_len requested,
    and tranformes the data to fit to RF (temporal-hot encodings and exams avg per year etc).
    :param train_window_len: mask all patients events succeeding train_window_len
    :param trajectory_length_threshold: Filter patients with trajectory_length >= trajectory_length_threshold
    :return: df_events_agg: the events counters for the relevant patients history
    '''
    # Mask features by train_window_len
    date_cols = [col for col in cat_based_df.columns if 'date' in col]
    masked_cat_based_df = cat_based_df.copy()
    # Mask Dates and change format if needed
    for col in date_cols:
        masked_cat_based_df[col] = masking_rule(masked_cat_based_df, col, train_window_len, intervals_from_diagnosis=True)

    # Recalculate duration based on next drug start (if exists), otherwise based on last available month
    mask_drug_durations(train_window_len, masked_cat_based_df)

    # Mask events
    for i, date_col in enumerate(date_cols):

        if date_col in GENERAL_DATE_EVENT_MAPPING:
            # Filter patients with zero medical intervention - TODO: uncomment if fillna!
            # masked_cat_based_df = filter_zero_medical_intervention_patients(masked_cat_based_df)
            # leave date column, no event column to mask.
            if date_col=='diagnosis_date':
                # No need of these columns since it's aligned for all patients and set to 0
                masked_cat_based_df = masked_cat_based_df.drop(columns=[date_col])
            if date_col not in ['birth_date', 'death_date']:
                masked_cat_based_df[date_col] = masked_cat_based_df[date_col].replace({np.nan: np.finfo(np.float32).min})

        elif date_col in PER_PATIENT_DATE_EVENT_MAPPING:
            # Mask event values columns based on corresponding date columns
            event_col = PER_PATIENT_DATE_EVENT_MAPPING[date_col][1]
            masked_cat_based_df[event_col] = masked_cat_based_df[event_col].where(masked_cat_based_df[date_col].notna(), np.nan)
            # Filter patients with zero medical intervention - TODO: uncomment if fillna!
            # masked_cat_based_df = filter_zero_medical_intervention_patients(masked_cat_based_df)

            if masked_cat_based_df[event_col].dtype == 'object':
                # 1. Create one hot encodings to the event_col
                event_temporal_hot = pd.get_dummies(masked_cat_based_df[event_col], prefix=event_col, dummy_na=False).astype('int')
                # 2. take corresponding date_col, and for each patient where {event_col}__{value} is 1->
                # replace with the patient's date_col value
                for col in event_temporal_hot.columns:
                    if '_nan' not in col.lower():
                        event_temporal_hot[col] = event_temporal_hot[col].replace({0: np.finfo(np.float32).min})
                        event_temporal_hot[col] = event_temporal_hot[col].where(
                            event_temporal_hot[col] != 1,
                            masked_cat_based_df.loc[event_temporal_hot.index, date_col]
                        )
                    # else: leave dummyna columns unchanged
                # 3. add the temporal hot encodings with existing event and date columns
                masked_cat_based_df = pd.concat([masked_cat_based_df, event_temporal_hot], axis=1)
        else:
            print(f"date column: {date_col} not mapped in mappers")

    # transform to avg. per year. -100 if missing.
    masked_cat_based_df = transform_exams_to_avg_per_year(masked_cat_based_df, train_window_len)
    # Take the latest categorical event of each value series of temporal-hot encodings
    masked_cat_based_df = create_latest_event_value_encoding(masked_cat_based_df)

    # Filter patients with trajectory_length >= threshold
    masked_cat_based_df = masked_cat_based_df[masked_cat_based_df["end_of_data"] + 1 >= trajectory_length_threshold]
    # masked_cat_based_df = masked_cat_based_df.set_index('patient_id')

    # # Filter patients with zero medical intervention  - TODO: uncomment if not fillna! (since zero patients are discovered by counting nans)
    masked_cat_based_df = filter_zero_medical_intervention_patients(masked_cat_based_df)

    return masked_cat_based_df


def mask_cat_based_data_for_catboost(cat_based_df:pd.DataFrame, train_window_len:int, trajectory_length_threshold:int) -> pd.DataFrame:
    '''
    This function masks the events according to the train_window_len requested,
    and tranformes the data to fit to RF (temporal-hot encodings and exams avg per year etc).
    :param train_window_len: mask all patients events succeeding train_window_len
    :param trajectory_length_threshold: Filter patients with trajectory_length >= trajectory_length_threshold
    :return: df_events_agg: the events counters for the relevant patients history
    '''
    # Mask features by train_window_len
    date_cols = [col for col in cat_based_df.columns if 'date' in col]
    masked_cat_based_df = cat_based_df.copy()
    # Mask Dates and change format if needed
    for col in date_cols:
        masked_cat_based_df[col] = masking_rule(masked_cat_based_df, col, train_window_len, intervals_from_diagnosis=True)

    # Recalculate duration based on next drug start (if exists), otherwise based on last available month
    mask_drug_durations(train_window_len, masked_cat_based_df)

    # Mask events
    for i, date_col in enumerate(date_cols):

        if date_col in GENERAL_DATE_EVENT_MAPPING:
            # leave date column, no event column to mask.
            if date_col=='diagnosis_date':
                # No need of these columns since it's aligned for all patients and set to 0
                masked_cat_based_df = masked_cat_based_df.drop(columns=[date_col])

        elif date_col in PER_PATIENT_DATE_EVENT_MAPPING:
            # Mask event values columns based on corresponding date columns
            event_col = PER_PATIENT_DATE_EVENT_MAPPING[date_col][1]
            masked_cat_based_df[event_col] = masked_cat_based_df[event_col].where(masked_cat_based_df[date_col].notna(), np.nan)
        else:
            print(f"date column: {date_col} not mapped in mappers")

    # Fill categorical NaN with 'missing' label for CatBoost
    cat_features = [col for i, col in enumerate(masked_cat_based_df.columns) if masked_cat_based_df[col].dtype == "object"]
    for col in cat_features:
        masked_cat_based_df[col] = masked_cat_based_df[col].astype(object).where(masked_cat_based_df[col].notna(), 'missing').astype(str)

    # Filter patients with trajectory_length >= threshold
    masked_cat_based_df = masked_cat_based_df[masked_cat_based_df["end_of_data"] + 1 >= trajectory_length_threshold]
    # masked_cat_based_df = masked_cat_based_df.set_index('patient_id')

    # Filter patients with zero medical intervention
    masked_cat_based_df = filter_zero_medical_intervention_patients(masked_cat_based_df)

    return masked_cat_based_df


def mask_cat_based_data_for_transformer(cat_based_df: pd.DataFrame, train_window_len: int,
                                     trajectory_length_threshold: int) -> pd.DataFrame:
    '''
    This function masks the events according to the train_window_len requested,
    and tranformes the data to fit to RF (temporal-hot encodings and exams avg per year etc).
    :param train_window_len: mask all patients events succeeding train_window_len
    :param trajectory_length_threshold: Filter patients with trajectory_length >= trajectory_length_threshold
    :return: df_events_agg: the events counters for the relevant patients history
    '''
    # Mask features by train_window_len
    date_cols = [col for col in cat_based_df.columns if 'date' in col]
    masked_cat_based_df = cat_based_df.copy()
    # Mask Dates and change format if needed
    for col in date_cols:
        masked_cat_based_df[col] = masking_rule(masked_cat_based_df, col, train_window_len,
                                                intervals_from_diagnosis=True)

    # Recalculate duration based on next drug start (if exists), otherwise based on last available month
    mask_drug_durations(train_window_len, masked_cat_based_df)

    # Mask events
    for i, date_col in enumerate(date_cols):

        if date_col in GENERAL_DATE_EVENT_MAPPING:
            # leave date column, no event column to mask.
            if date_col == 'diagnosis_date':
                # No need of these columns since it's aligned for all patients and set to 0
                masked_cat_based_df = masked_cat_based_df.drop(columns=[date_col])

        elif date_col in PER_PATIENT_DATE_EVENT_MAPPING:
            # Mask event values columns based on corresponding date columns
            event_col = PER_PATIENT_DATE_EVENT_MAPPING[date_col][1]
            masked_cat_based_df[event_col] = masked_cat_based_df[event_col].where(masked_cat_based_df[date_col].notna(),
                                                                                  np.nan)
        else:
            print(f"date column: {date_col} not mapped in mappers")

    # Filter patients with trajectory_length >= threshold
    masked_cat_based_df = masked_cat_based_df[masked_cat_based_df["end_of_data"] + 1 >= trajectory_length_threshold]
    # masked_cat_based_df = masked_cat_based_df.set_index('patient_id')

    # Filter patients with zero medical intervention
    masked_cat_based_df = filter_zero_medical_intervention_patients(masked_cat_based_df)

    return masked_cat_based_df

def compute_targets(cat_based_df: pd.DataFrame, df_events_agg: pd.DataFrame, train_window_len: int, followup_months=6):
    """
    Computes various target metrics for patient data within a specified follow-up period.

    Parameters:
    - cat_based_df: DataFrame containing the full patient data in categorical format.
    - df_events_agg: DataFrame with aggregated event data, indexed similarly to cat_based_df.
    - train_train_window_len: The last month index in the training set; targets are computed from train_train_window_len+1 to train_train_window_len+followup_months.
    - followup_months: Number of months to consider for computing the targets.

    Returns:
    - A dictionary with computed target metrics.
    """
    # Define drug and procedure columns
    drug_columns = BIOLOGIC_DRUG_COLUMNS + SMALL_MOLECULE_DRUG_COLUMNS
    procedure_columns = PROCEDURE_COLUMNS
    ibd_related_feature_cols = drug_columns + procedure_columns

    # Define the follow-up window
    followup_start = train_window_len
    followup_end = train_window_len + followup_months - 1

    # Initialize target dictionary
    target_dict = {}

    # Compute target_drug_switch: 1 if any drug event occurs in the follow-up window, else 0
    target_dict['target_drug_switch'] = cat_based_df.loc[df_events_agg.index, drug_columns].apply(
        lambda row: row.between(followup_start, followup_end).any(), axis=1).astype(int)

    # Compute target_procedure: 1 if any procedure event occurs in the follow-up window, else 0
    target_dict['target_procedure'] = cat_based_df.loc[df_events_agg.index, procedure_columns].apply(
        lambda row: row.between(followup_start, followup_end).any(), axis=1).astype(int)

    # Compute target_event_count: total number of IBD-related events in the follow-up window
    target_dict['target_event_count'] = cat_based_df.loc[df_events_agg.index, ibd_related_feature_cols].apply(
        lambda row: row.between(followup_start, followup_end).sum(), axis=1)

    # Compute target_time_to_next_switch_within_window: time to next drug switch within the follow-up window
    target_dict['target_time_to_next_switch_within_window'] = cat_based_df.loc[df_events_agg.index, drug_columns].apply(
        lambda row: (row[row.between(followup_start, followup_end)].min() - followup_start) if row.between(followup_start, followup_end).any() else -100,
        axis=1)

    # Compute target_time_to_next_switch: time to next drug switch after the training period
    def compute_time_to_next_switch(row):
        # Determine the patient's trajectory length
        trajectory_length = cat_based_df.loc[row.name, 'end_of_data']+1
        # Define the window from follow-up_start to trajectory_length
        window_end = trajectory_length
        # Filter events in the window
        future_events = row[row.between(followup_start, window_end)]
        if not future_events.empty:
            return future_events.min() - followup_start
        else:
            return -100

    target_dict['target_time_to_next_switch'] = cat_based_df.loc[df_events_agg.index, drug_columns].apply(
        compute_time_to_next_switch, axis=1)

    return target_dict
