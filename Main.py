import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.api import VAR
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import braycurtis
from sklearn.preprocessing import LabelEncoder
import time
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# To ignore specific types of warnings (for example, ConvergenceWarnings from ARIMAX)
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Load data
train_metadata = pd.read_csv('train_metadata.csv')
train_data = pd.read_csv('train_data.csv')
test_metadata = pd.read_csv('test_metadata.csv')
short_timeseries_data = pd.read_csv('short_timeseries_data.csv')

np.random.seed(42)



def interpolate_with_random_forest(metadata, data):
    regr = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    complete_data = data.copy()

    # Step 1: Encode baboon_id as a numeric feature
    metadata = metadata.copy()
    label_encoder = LabelEncoder()
    metadata['baboon_id_encoded'] = label_encoder.fit_transform(metadata['baboon_id'])

    # Step 2: Normalize metadata features (if needed)
    feature_cols = metadata.columns.difference(['sample', 'collection_date', 'baboon_id'])
    metadata[feature_cols] = metadata[feature_cols].apply(lambda x: (x - x.mean()) / x.std(), axis=0)

    # Step 3: Train on known data
    y = data.drop(columns=['sample']).dropna()
    X = metadata[feature_cols].loc[y.index]
    regr.fit(X, y)

    # Step 5: Interpolate synthetic samples
    missing_samples = metadata[~metadata['sample'].isin(data['sample'])]
    missing_samples = missing_samples[~missing_samples['sample'].isin(complete_data['sample'])]
    if not missing_samples.empty:
        missing_X = missing_samples[feature_cols]
        predicted_values = regr.predict(missing_X)
        predicted_df = pd.DataFrame(predicted_values, columns=y.columns)
        predicted_df['sample'] = missing_samples['sample'].values
        complete_data = pd.concat([complete_data, predicted_df], ignore_index=True, sort=False)

    # Step 6: Normalize predicted microbiome compositions
    target_columns = complete_data.columns.difference(['sample'])
    row_sums = complete_data[target_columns].sum(axis=1)
    complete_data[target_columns] = complete_data[target_columns].div(row_sums, axis=0)

    return complete_data


# Step 2: Create cross-validation splits
def monkey_based_split(metadata, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(metadata, metadata['baboon_id']):
        yield train_idx, val_idx


# Step 3: Fill missing dates
def fill_missing_dates(train_metadata, test_metadata=None):
    train_metadata['collection_date'] = pd.to_datetime(train_metadata['collection_date'])
    if test_metadata is not None:
        test_metadata['collection_date'] = pd.to_datetime(test_metadata['collection_date'])
    all_filled_rows = []

    for baboon_id, group in train_metadata.groupby('baboon_id'):
        group = group.sort_values(by='collection_date').reset_index(drop=True)
        min_date, max_date = group['collection_date'].min(), group['collection_date'].max()

        if test_metadata is not None:
            input_min_date = test_metadata[test_metadata['baboon_id'] == baboon_id]['collection_date'].min()
            max_date = min(max_date, input_min_date - pd.Timedelta(days=1)) if not pd.isna(input_min_date) else max_date
        
        all_dates = pd.date_range(start=min_date, end=max_date, freq='D')

        # Create a DataFrame for all dates
        all_dates_df = pd.DataFrame({'collection_date': all_dates})
        merged = pd.merge(all_dates_df, group, on='collection_date', how='left')

        # Fill in missing values for metadata attributes
        merged['baboon_id'] = baboon_id
        merged['sample'] = merged['sample'].fillna(
            merged['collection_date'].dt.strftime(f"synthetic_{baboon_id}_%Y%m%d")
        )
        merged['sex'] = merged['sex'].fillna(method='ffill').fillna(method='bfill')
        merged['age'] = merged['age'].interpolate(
            method='linear', limit_direction='both'
        )
        merged['social_group'] = merged['social_group'].fillna(method='ffill').fillna(method='bfill')
        merged['group_size'] = merged['group_size'].fillna(method='ffill').fillna(method='bfill')
        merged['rain_month_mm'] = merged['rain_month_mm'].fillna(method='ffill').fillna(method='bfill')

        merged['season'] = merged['collection_date'].dt.month.apply(
            lambda m: 'dry' if 6 <= m <= 10 else 'wet'
        )
        merged['hydro_year'] = merged['collection_date'].dt.year + (merged['collection_date'].dt.month >= 11).astype(
            int)

        # Add the 'month' column based on the 'collection_date'
        merged['month'] = merged['collection_date'].dt.month

        # Impute diet_PC values using linear regression
        diet_cols = [f'diet_PC{i}' for i in range(1, 14)]
        valid_diet_data = group[['collection_date'] + diet_cols].dropna()
        if not valid_diet_data.empty:
            X = (valid_diet_data['collection_date'] - valid_diet_data['collection_date'].min()).dt.days.values.reshape(
                -1, 1)
            models = {col: LinearRegression().fit(X, valid_diet_data[col]) for col in diet_cols}
            missing_idx = merged[diet_cols[0]].isna()
            missing_dates = merged.loc[missing_idx, 'collection_date']
            if not missing_dates.empty:
                days_from_min = (missing_dates - valid_diet_data['collection_date'].min()).dt.days.values.reshape(-1, 1)
                for col in diet_cols:
                    merged.loc[missing_idx, col] = models[col].predict(days_from_min)

        all_filled_rows.append(merged)

    filled_metadata = pd.concat(all_filled_rows, ignore_index=True)

    filled_metadata['sex'] = filled_metadata['sex'].map({'F': 1, 'M': 0})
    filled_metadata['season'] = filled_metadata['season'].map({'dry': 1, 'wet': 0})
    filled_metadata['social_group'] = filled_metadata['social_group'].str.replace('g_', '').astype(float)
    return filled_metadata


# Step 5: Interpolate missing data using RandomForest
def interpolate_with_random_forest(metadata, data):
    regr = RandomForestRegressor(n_estimators=100, n_jobs=-1)
    complete_data = data.copy()
    feature_cols = metadata.columns.difference(['sample', 'collection_date', 'baboon_id'])
    y = data.drop(columns=['sample'])
    X = metadata[feature_cols].loc[y.index]
    regr.fit(X, y)

    missing_samples = metadata[metadata['sample'].str.startswith('synthetic')] #metadata[~metadata['sample'].isin(data['sample'])]
    if not missing_samples.empty:
        missing_X = missing_samples[feature_cols]
        predicted_values = regr.predict(missing_X)
        predicted_df = pd.DataFrame(predicted_values, columns=y.columns)
        predicted_df['sample'] = missing_samples['sample'].values
        complete_data = pd.concat([complete_data, predicted_df], ignore_index=True)

    return complete_data


# Step 6: Predict future samples with VAR
def var_predict(data):
    model = VAR(data.set_index('sample'))
    results = model.fit()
    return results


# Main pipeline
def main(train_metadata, train_data, test_metadata, short_timeseries_data, n_splits=5):
    fold_results = []
    for fold_num, (train_idx, val_idx) in enumerate(monkey_based_split(train_metadata, n_splits)):
        start_time = time.time()

        # Step 3: Fill missing dates
        val_fold_metadata = train_metadata.iloc[val_idx]
        train_fold_metadata = fill_missing_dates(train_metadata.iloc[train_idx], val_fold_metadata)
        val_fold_metadata = fill_missing_dates(val_fold_metadata)

        # Process the train_metadata DataFrame


        # Step 5: Interpolation
        interpolated_train_data = interpolate_with_random_forest(train_fold_metadata, train_data)
        interpolated_val_data = interpolate_with_random_forest(val_fold_metadata, train_data)

        # Step 6: Train model
        model = var_predict(interpolated_train_data)

        # Step 7: Cross-validation
        # Forecast only for samples that exist in metadata but not in data (missing microbiome composition)
        val_missing_samples = val_fold_metadata[~val_fold_metadata['sample'].isin(train_data['sample'])]
        if not val_missing_samples.empty:
            val_missing_features = val_missing_samples.set_index('sample').values
            val_predictions = model.forecast(interpolated_train_data.set_index('sample').values, steps=len(val_missing_features))

            # Efficient calculation of Bray-Curtis distances using batching to avoid memory errors
            batch_size = 1000
            total_bc_score = 0
            num_batches = 0
            for start_idx in range(0, len(val_predictions), batch_size):
                end_idx = min(start_idx + batch_size, len(val_predictions))
                batch_val_predictions = val_predictions[start_idx:end_idx]
                batch_val_data = interpolated_val_data.set_index('sample').values[start_idx:end_idx]
                bc_scores = np.array([braycurtis(a, b) for a, b in zip(batch_val_predictions, batch_val_data)]).mean().mean()
                total_bc_score += bc_scores
                num_batches += 1

            average_bc_score = total_bc_score / num_batches

            fold_results.append(average_bc_score)

    # Step 8: Print the mean BC score
    mean_bc_score = np.mean(fold_results)
    print(f'Mean BC score for all folds: {mean_bc_score:.4f}')

    # Step 9: Test the model on the test data
    combined_metadata = pd.concat([train_metadata, test_metadata], ignore_index=True)
    combined_data = pd.concat([train_data, short_timeseries_data], ignore_index=True)

    test_missing_samples = combined_metadata[~combined_metadata['sample'].isin(short_timeseries_data['sample'])]

    test_metadata_filled = fill_missing_dates(combined_metadata)

    # Ensure interpolated_test_data is used for test_missing_samples preparation
    interpolated_test_data = interpolate_with_random_forest(test_metadata_filled, combined_data)

    if not test_missing_samples.empty:
        # Get starting points for forecasting from the interpolated test data
        last_known_values = interpolated_test_data.set_index('sample').values
        num_missing_samples = len(test_missing_samples)

        # Forecast for the missing samples
        test_predictions = model.forecast(last_known_values, steps=num_missing_samples)

        predictions_df = pd.DataFrame(
            test_predictions,
            columns=short_timeseries_data.drop(columns=['sample']).columns
        )
        predictions_df['sample'] = test_missing_samples['sample'].values

        # Reorder columns to make 'sample' the first column
        columns = ['sample'] + [col for col in predictions_df.columns if col != 'sample']
        predictions_df = predictions_df[columns]
        
        predictions_df = predictions_df[~predictions_df['sample'].isin(train_data['sample'])]
        
        # Save predictions
        predictions_df.to_csv('test_predictions.csv', index=False)

main(train_metadata, train_data, test_metadata, short_timeseries_data)

# Display final output to user
print(pd.read_csv('test_predictions.csv'))
