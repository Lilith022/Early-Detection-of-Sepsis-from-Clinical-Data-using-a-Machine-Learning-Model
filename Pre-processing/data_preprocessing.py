import pandas as pd
import numpy as np
import os

def forward_fill_laboratorio(df):
    """
    Forward-fill laboratory test values with a maximum validity window of 12 hours.

    Rules:
    - If the first non-missing measurement appears after the start of the record, its value is also 
      propagated backward up to 12 hours.
    - For subsequent measurements, values are forward-filled up to 12 hours after the last observed value.
    """
    lab_columns = df.loc[:, "BaseExcess":"Platelets"].columns.tolist()

    for column in lab_columns:
        if column in df.columns:
            original_series = df[column].values
            new_series = original_series.copy()

            first_valid_idx = None
            for i, val in enumerate(original_series):
                if not pd.isna(val):
                    first_valid_idx = i
                    break

            if first_valid_idx is not None and first_valid_idx > 0:
                start_fill = max(0, first_valid_idx - 12)
                new_series[start_fill:first_valid_idx] = original_series[first_valid_idx]

            last_real = None
            count_since_real = 0
            for i in range(len(new_series)):
                if not pd.isna(new_series[i]):
                    last_real = new_series[i]
                    count_since_real = 0
                elif last_real is not None and count_since_real < 12:
                    new_series[i] = last_real
                    count_since_real += 1

            df[column] = new_series
    return df

def select_time_segment(df, patient_id):
    """
    Extracts a fixed 21-hour observation window for each patient.

    - Septic: aligned to the first positive SepsisLabel (requires ≥11 hours before and ≥6 after). 
      If ≥9 after, uses a full 21-hour window; otherwise pads with NaNs.
    - Non-septic: excluded if <18 total hours. Otherwise, selects the 21-hour window 
      with highest data density, or pads if length <21.
    """

    categorical_vars = ["Age", "Gender", "SepsisLabel"]
    iculos_var = "ICULOS"

    labels = df["SepsisLabel"].values
    is_septic = labels.max() == 1 

    if is_septic:
        first_sepsis_idx = np.where(labels == 1)[0][0]
        previous_time = first_sepsis_idx >= 11
        time_after = len(df) - first_sepsis_idx - 1 >= 6  
        full_window = len(df) - first_sepsis_idx - 1 >= 9  

        if not (previous_time and time_after):
            return None
        if full_window:
            start = first_sepsis_idx - 11
            end = first_sepsis_idx + 10  
            window_df = df.iloc[start:end + 1].copy() 
        else:
            start = first_sepsis_idx - 11
            window_df = df.iloc[start:].copy()
            missing = 21 - len(window_df)
            padding = pd.DataFrame(np.nan, index=range(missing), columns=df.columns)  

            for col in categorical_vars:
                padding[col] = window_df[col].iloc[-1]
            last_iculos = window_df[iculos_var].iloc[-1]
            padding[iculos_var] = np.arange(last_iculos + 1, last_iculos + 1 + missing)
            window_df = pd.concat([window_df, padding], ignore_index=True)

    else:
        if len(df) < 18:
            return None
        if len(df) >= 21:
            best_window = None
            best_nonnan = -1
            for start in range(len(df) - 20):  
                window = df.iloc[start:start + 21]  
                total_nans = window.notna().sum().sum()
                if total_nans > best_nonnan:
                    best_window = window 
                    best_nonnan = total_nans
            window_df = best_window.copy()
        else:
            window_df = df.copy()
            missing = 21 - len(window_df)
            padding = pd.DataFrame(np.nan, index=range(missing), columns=df.columns)

            for col in categorical_vars:
                padding[col] = window_df[col].iloc[-1]
            last_iculos = window_df[iculos_var].iloc[-1]
            padding[iculos_var] = np.arange(last_iculos + 1, last_iculos + 1 + missing)
            window_df = pd.concat([window_df, padding], ignore_index=True)

    window_df.insert(0, "Paciente", patient_id)
    return window_df

def calculate_hr_sbp(df):
    """Compute the HR/SBP ratio as a clinical feature"""

    if "HR" in df.columns and "SBP" in df.columns:
        df.loc[:, "HR_SBP"] = df["HR"] / df["SBP"]
        cols = ["Paciente", "HR_SBP"] + [col for col in df.columns if col not in ["Paciente", "HR_SBP"]]
        df = df[cols]
    return df

def patients_without_data(df):
    """
    Remove patients with no available data in at least one variable group
    (vital signs, laboratory, or categorical). 
    """

    cols = df.columns.tolist()
    signs_cols = cols[cols.index("HR_SBP"): cols.index("EtCO2") + 1]
    lab_cols    = cols[cols.index("BaseExcess"): cols.index("Platelets") + 1]
    categ_cols  = cols[cols.index("Age"): cols.index("SepsisLabel") + 1]
    valid_patients = []
    removed_patients = []

    for patient_id, patient_df in df.groupby("Paciente"):
        no_signs = patient_df[signs_cols].isna().all().all()
        no_labs = patient_df[lab_cols].isna().all().all()
        no_categ = patient_df[categ_cols].isna().all().all()

        if not (no_signs or no_labs or no_categ):
            valid_patients.append(patient_df)
        else:
            removed_patients.append(patient_id)

    df_filtrado = pd.concat(valid_patients, ignore_index=True)
    return df_filtrado

def sliding_windows(df):
    """
    Extracts sliding 6-hour windows per patient, computing statistical features
    for vital signs, laboratory, and categorical variables.
    """
    vital_signs = df.loc[:, "HR_SBP":"EtCO2"].columns.tolist()   
    lab_vars = df.loc[:, "BaseExcess":"Platelets"].columns.tolist()  
    categorical_vars = df.loc[:, "Age":"SepsisLabel"].columns.tolist() 
    results = []

    for patient_id, patient_df in df.groupby("Paciente"):
        patient_df = patient_df.reset_index(drop=True)

        for start in range(0, 16): 
            window = patient_df.iloc[start:start+6]
            row_result = {"Paciente": patient_id}

            for col in vital_signs:
                row_result[f"{col}_min"] = window[col].min(skipna=True)
                row_result[f"{col}_max"] = window[col].max(skipna=True)
                row_result[f"{col}_mean"] = window[col].mean(skipna=True)
                row_result[f"{col}_var"] = window[col].var(skipna=True, ddof=1) 
                row_result[f"{col}_last"] = window[col].iloc[-1]

            for col in lab_vars:
                row_result[f"{col}_min"] = window[col].min(skipna=True)
                row_result[f"{col}_max"] = window[col].max(skipna=True)

            for col in categorical_vars:
                row_result[col] = window[col].iloc[-1]

            results.append(row_result)
    results_df = pd.DataFrame(results)
    return results_df

import pandas as pd

def remove_high_nan_columns(df, threshold=0.2):
    """
    Remove features from the DataFrame where the proportion of missing values exceeds 20%.
    """
    nan_fraction = df.isna().mean()  
    cols_to_drop = nan_fraction[nan_fraction > threshold].index.tolist()
    return df.drop(columns=cols_to_drop)


if __name__ == "__main__":
    
    input_folders = ["D:/Desktop/2025-1/TRABAJO DE GRADO 1/physionet.org/files/challenge-2019/1.0.0/training/training_Pocos_archivos/training_setA",
                     "D:/Desktop/2025-1/TRABAJO DE GRADO 1/physionet.org/files/challenge-2019/1.0.0/training/training_Pocos_archivos/training_setB"]
    output_file = "D:/Desktop/Tesis/Data/Resultados/pre_imputation_dataset.psv" 
    all_results = []
    cols_to_remove = ["Unit1", "Unit2", "HospAdmTime"]

    def process_file(file_path, patient_id):
        df = pd.read_csv(file_path, sep="|")
        df = df.drop(columns=[col for col in cols_to_remove if col in df.columns], errors="ignore")
        
        for func in [
            forward_fill_laboratorio,
            lambda x: select_time_segment(x, patient_id),
            calculate_hr_sbp,
            patients_without_data,
            sliding_windows
        ]:
            df = func(df)
            if df is None:
                return None
        return df
    
    for folder in input_folders:
        for filename in os.listdir(folder):
            if filename.endswith(".psv"):
                file_path = os.path.join(folder, filename)
                patient_id = filename.replace(".psv", "")
                final_df = process_file(file_path, patient_id)
                if final_df is None:
                    continue
                all_results.append(final_df)
    dataset = pd.concat(all_results, ignore_index=True)
    dataset = remove_high_nan_columns(dataset, threshold=0.2) 
    dataset.to_csv(output_file, sep="|", index=False)


