import pandas as pd
import numpy as np
import joblib
import os
import sqlite3
from datetime import timedelta, datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from statsmodels.tsa.arima.model import ARIMA
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)

# File paths (relative to project root)
SQLITE_DB_PATH = '../historic_data.db'
IF_MODEL_PATH = '../isolation_forest_model.pkl'
KMEANS_MODEL_PATH = '../kmeans_model.pkl'
LOF_MODEL_PATH = '../lof_model.pkl'
ENCODER_PATH = '../encoder.pkl'
SCALER_PATH = '../scaler.pkl'

def create_db_table_if_not_exists(conn):
    """Create SQLite table if it doesn't exist."""
    try:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS file_activity (
                User TEXT,
                Filename TEXT,
                Process TEXT,
                Activity TEXT,
                Time_Accessed TEXT
            )
        ''')
        conn.commit()
        logger.info("SQLite table created or verified.")
    except sqlite3.Error as e:
        logger.error(f"Error creating SQLite table: {e}")
        raise

def load_history_data():
    """Load historical data from SQLite."""
    if os.path.exists(SQLITE_DB_PATH):
        try:
            conn = sqlite3.connect(SQLITE_DB_PATH)
            create_db_table_if_not_exists(conn)
            df = pd.read_sql_query("SELECT * FROM file_activity", conn)
            conn.close()
            df['Time_Accessed'] = pd.to_datetime(df['Time_Accessed'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            df = df.dropna(subset=['Time_Accessed'])
            logger.info(f"Historic data loaded, {len(df)} records.")
            return df
        except (sqlite3.Error, pd.errors.EmptyDataError) as e:
            logger.error(f"Error loading historic data: {e}")
            return pd.DataFrame()
    logger.info("No historic data found, starting fresh.")
    return pd.DataFrame()

def save_history_data(df, retention_days=30):
    """Save data to SQLite with retention policy."""
    df = df.copy()
    df['Time_Accessed'] = pd.to_datetime(df['Time_Accessed'], errors='coerce')
    df = df[df['Time_Accessed'] >= datetime.now() - timedelta(days=retention_days)]
    
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        create_db_table_if_not_exists(conn)
        conn.execute("DELETE FROM file_activity")
        df['Time_Accessed'] = df['Time_Accessed'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df.to_sql('file_activity', conn, if_exists='append', index=False)
        conn.commit()
        conn.close()
        logger.info(f"Historic data saved, {len(df)} records (retained last {retention_days} days).")
    except sqlite3.Error as e:
        logger.error(f"Error saving historic data: {e}")
        raise

def load_model_artifacts():
    """Load saved models, encoder, and scaler."""
    models = {'isolation_forest': None, 'kmeans': None, 'lof': None}
    encoder, scaler = None, None
    try:
        if os.path.exists(IF_MODEL_PATH):
            models['isolation_forest'] = joblib.load(IF_MODEL_PATH)
            logger.info("Isolation Forest model loaded.")
        if os.path.exists(KMEANS_MODEL_PATH):
            models['kmeans'] = joblib.load(KMEANS_MODEL_PATH)
            logger.info("K-Means model loaded.")
        if os.path.exists(LOF_MODEL_PATH):
            models['lof'] = joblib.load(LOF_MODEL_PATH)
            logger.info("LOF model loaded.")
        if os.path.exists(ENCODER_PATH):
            encoder = joblib.load(ENCODER_PATH)
            logger.info("Encoder loaded.")
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            logger.info("Scaler loaded.")
    except Exception as e:
        logger.error(f"Error loading model artifacts: {e}")
    return models, encoder, scaler

def save_model_artifacts(models, encoder, scaler):
    """Save models, encoder, and scaler."""
    try:
        joblib.dump(models['isolation_forest'], IF_MODEL_PATH)
        joblib.dump(models['kmeans'], KMEANS_MODEL_PATH)
        joblib.dump(models['lof'], LOF_MODEL_PATH)
        joblib.dump(encoder, ENCODER_PATH)
        joblib.dump(scaler, SCALER_PATH)
        logger.info("Models, encoder, and scaler saved.")
    except Exception as e:
        logger.error(f"Error saving model artifacts: {e}")
        raise

def preprocess_logs(df, encoder=None, scaler=None, is_training=True):
    """Preprocess data for modeling, including time-series and peer group features."""
    try:
        df = df.dropna()
        df['Time_Accessed'] = pd.to_datetime(df['Time_Accessed'], format='%d-%m-%Y %H:%M', errors='coerce')
        df = df.dropna(subset=['Time_Accessed'])
        df['Hour'] = df['Time_Accessed'].dt.hour
        df['DayOfWeek'] = df['Time_Accessed'].dt.dayofweek
        df['FileType'] = df['Filename'].apply(lambda x: os.path.splitext(x)[1].lower() or '.unknown')
        df['Directory'] = df['Filename'].apply(lambda x: os.path.dirname(x))

        df['LargeFileAccessCount'] = 0
        large_file_types = ['.xlsx', '.zip', '.csv', '.mp4', '.docx']
        df.sort_values(by='Time_Accessed', inplace=True)
        for user in df['User'].unique():
            user_df = df[df['User'] == user]
            large_files = user_df[user_df['FileType'].isin(large_file_types)]
            for idx, current_time in large_files['Time_Accessed'].items():
                time_window = (df['User'] == user) & \
                              (df['FileType'].isin(large_file_types)) & \
                              (df['Time_Accessed'].between(current_time - timedelta(minutes=1), current_time + timedelta(minutes=1)))
                count = df[time_window].shape[0]
                df.at[idx, 'LargeFileAccessCount'] = count

        df['UserAccessCount'] = df['User'].map(df['User'].value_counts())
        df['FileAccessCount'] = df['Filename'].map(df['Filename'].value_counts())

        user_activity_count = df.groupby(['User', 'Activity']).size().unstack(fill_value=0)
        user_activity_count.columns = [f'Activity_{col}_Count' for col in user_activity_count.columns]
        df = df.merge(user_activity_count, left_on='User', right_index=True, how='left')

        categorical_columns = ['User', 'Process', 'Activity', 'FileType', 'Directory']
        if is_training:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded_data = encoder.fit_transform(df[categorical_columns])
        else:
            if encoder is None:
                raise ValueError("Encoder must be provided for inference.")
            encoded_data = encoder.transform(df[categorical_columns])

        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns))
        modeling_df = pd.concat([df.drop(columns=categorical_columns + ['Time_Accessed', 'Filename']), encoded_df], axis=1)

        numerical_columns = ['Hour', 'DayOfWeek', 'UserAccessCount', 'FileAccessCount', 'LargeFileAccessCount'] + \
                            [col for col in modeling_df.columns if 'Activity_' in col and '_Count' in col]
        if is_training:
            scaler = StandardScaler()
            modeling_df[numerical_columns] = scaler.fit_transform(modeling_df[numerical_columns])
        else:
            if scaler is None:
                raise ValueError("Scaler must be provided for inference.")
            modeling_df[numerical_columns] = scaler.transform(modeling_df[numerical_columns])

        logger.info("Data preprocessing completed.")
        return df, modeling_df, encoder, scaler
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise

def analyze_anomalies_by_feature(df, feature, feature_value_col, feature_stats):
    """Generate detailed anomaly reports with reasons."""
    try:
        anomalies = df[df['IsAnomaly'] == True].copy()
        report_columns = ['User', feature_value_col, 'Time_Accessed', 'Filename', 'Process', 'AnomalyScore']
        report = anomalies[report_columns].copy()
        reasons = []
        sensitive_activities = ['Upload', 'Executed', 'Delete', 'share']

        for _, row in report.iterrows():
            value = row[feature_value_col]
            reason = []

            if feature_value_col in feature_stats:
                total_count = feature_stats[feature_value_col].get(value, 0)
                proportion = total_count / len(df)
                if proportion < 0.05:
                    reason.append(f"Rare {feature_value_col} (frequency: {proportion:.3f})")

            if feature == 'Activity':
                user_activities = df[df['User'] == row['User']]['Activity'].value_counts()
                if value in user_activities:
                    proportion = user_activities[value] / user_activities.sum()
                    if proportion < 0.1:
                        reason.append(f"Unusual activity for user (frequency: {proportion:.3f})")
                if value in sensitive_activities:
                    reason.append("Sensitive activity")

            if feature == 'Hour':
                hour = int(value)
                if hour < 9 or hour > 17:
                    reason.append("Unusual hour (outside work hours)")

            if 'LargeFileAccessCount' in row and row['LargeFileAccessCount'] > 1:
                reason.append(f"Rapid large file access detected ({int(row['LargeFileAccessCount'])} files)")

            reasons.append("; ".join(reason) or "Anomalous pattern in feature combination")

        report['Reason'] = reasons
        return report.sort_values(by=['User', 'AnomalyScore'], ascending=[True, False])
    except Exception as e:
        logger.error(f"Error analyzing anomalies: {e}")
        return pd.DataFrame()

def train_arima_model(df, user, feature, hours=240):
    """Train ARIMA model for time-series forecasting per user."""
    try:
        user_df = df[df['User'] == user].sort_values('Time_Accessed')
        if len(user_df) < hours / 24:
            logger.warning(f"Insufficient data for ARIMA model for user {user}.")
            return None
        user_df = user_df.set_index('Time_Accessed').resample('h')[feature].sum().fillna(0)
        model = ARIMA(user_df, order=(1, 1, 1)).fit()
        logger.info(f"ARIMA model trained for user {user}.")
        return model
    except Exception as e:
        logger.warning(f"Failed to train ARIMA for user {user}: {e}")
        return None

def detect_arima_anomalies(df, arima_models, feature, confidence_interval=0.95):
    """Detect anomalies using ARIMA forecasts."""
    try:
        df = df.copy()
        df['ARIMA_Anomaly'] = False
        df['ARIMA_AnomalyScore'] = 0.0
        df['Hourly_Time'] = df['Time_Accessed'].dt.floor('h')  # Round to nearest hour

        for user in df['User'].unique():
            if user in arima_models and arima_models[user] is not None:
                user_df = df[df['User'] == user].sort_values('Time_Accessed')
                user_agg = user_df.set_index('Time_Accessed').resample('h')[feature].sum().fillna(0)
                if len(user_agg) > 0:
                    forecast = arima_models[user].forecast(steps=len(user_agg))
                    stderr = np.std(arima_models[user].resid)
                    threshold = stderr * 1.96  # 95% confidence interval
                    actual = user_agg.values
                    predicted = forecast.values
                    anomalies = np.abs(actual - predicted) > threshold
                    anomaly_scores = np.abs(actual - predicted)

                    # Create a DataFrame for hourly anomalies
                    anomaly_df = pd.DataFrame({
                        'Hourly_Time': user_agg.index,
                        'ARIMA_Anomaly': anomalies,
                        'ARIMA_AnomalyScore': anomaly_scores
                    })
                    anomaly_df['User'] = user

                    # Merge with original DataFrame
                    user_df = user_df.merge(
                        anomaly_df[['User', 'Hourly_Time', 'ARIMA_Anomaly', 'ARIMA_AnomalyScore']],
                        left_on=['User', 'Hourly_Time'],
                        right_on=['User', 'Hourly_Time'],
                        how='left'
                    )
                    user_df['ARIMA_Anomaly'] = user_df['ARIMA_Anomaly'].fillna(False)
                    user_df['ARIMA_AnomalyScore'] = user_df['ARIMA_AnomalyScore'].fillna(0.0)

                    # Update original DataFrame
                    df.loc[df['User'] == user, 'ARIMA_Anomaly'] = user_df['ARIMA_Anomaly'].values
                    df.loc[df['User'] == user, 'ARIMA_AnomalyScore'] = user_df['ARIMA_AnomalyScore'].values

        df = df.drop(columns=['Hourly_Time'])
        logger.info("ARIMA anomaly detection completed.")
        return df
    except Exception as e:
        logger.error(f"Error in ARIMA anomaly detection: {e}")
        return df

def periodic_retrain(new_data_path, contamination=0.05, max_history=100000, retention_days=30, min_groups=2, max_groups=10):
    """Main function for periodic retraining and anomaly detection."""
    logger.info("Starting advanced UBA retraining process.")

    historic_df = load_history_data()
    try:
        new_df = pd.read_csv(new_data_path)
        logger.info(f"New data loaded from {new_data_path} with {len(new_df)} records.")
        new_df['Time_Accessed'] = pd.to_datetime(new_df['Time_Accessed'], format='%d-%m-%Y %H:%M', errors='coerce')
        new_df = new_df.dropna(subset=['Time_Accessed'])
    except Exception as e:
        logger.error(f"Error loading new data: {e}")
        return historic_df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    models, encoder, scaler = load_model_artifacts()
    combined_df = pd.DataFrame()
    is_identical = False

    if not historic_df.empty:
        historic_df['Time_Accessed'] = pd.to_datetime(historic_df['Time_Accessed'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        historic_df = historic_df.dropna(subset=['Time_Accessed'])
        merged_df = pd.merge(
            new_df,
            historic_df,
            on=['User', 'Filename', 'Process', 'Activity', 'Time_Accessed'],
            how='left',
            indicator=True
        )
        if all(merged_df['_merge'] == 'both') and models['isolation_forest'] is not None:
            logger.info("New data is identical to historical data. Using existing models to generate anomaly reports.")
            combined_df = historic_df
            is_identical = True
        else:
            combined_df = pd.concat([historic_df, new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['User', 'Filename', 'Process', 'Activity', 'Time_Accessed'])
    else:
        combined_df = new_df
        logger.info("No historic data, training only on new data.")

    if len(combined_df) > max_history:
        combined_df = combined_df.sort_values('Time_Accessed').tail(max_history)
        logger.info(f"Truncated to {max_history} records.")

    original_df, modeling_df, encoder, scaler = preprocess_logs(combined_df, encoder, scaler, is_training=not is_identical)

    if not is_identical:
        models = {}
        logger.info("Training Isolation Forest...")
        models['isolation_forest'] = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
        models['isolation_forest'].fit(modeling_df)

        logger.info("Training K-Means...")
        n_clusters = min(max(min_groups, len(original_df['User'].unique()) // 5), max_groups)
        models['kmeans'] = KMeans(n_clusters=n_clusters, random_state=42)
        user_features = original_df.groupby('User')[['UserAccessCount', 'LargeFileAccessCount'] + 
                        [col for col in original_df.columns if 'Activity_' in col and '_Count' in col]].mean()
        if len(user_features) >= min_groups * 5:
            models['kmeans'].fit(user_features)
            user_groups = models['kmeans'].predict(user_features)
            original_df['PeerGroup'] = original_df['User'].map(dict(zip(user_features.index, user_groups)))
        else:
            logger.warning("Insufficient users for peer group clustering.")
            original_df['PeerGroup'] = 0
            models['kmeans'] = None

        logger.info("Training LOF...")
        models['lof'] = LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=True)
        if models['kmeans'] is not None:
            models['lof'].fit(user_features)

        logger.info("Training ARIMA models...")
        arima_models = {}
        for user in original_df['User'].unique():
            arima_models[user] = train_arima_model(original_df, user, 'LargeFileAccessCount')
    else:
        # For identical data, reapply K-Means clustering and ARIMA models
        user_features = original_df.groupby('User')[['UserAccessCount', 'LargeFileAccessCount'] + 
                        [col for col in original_df.columns if 'Activity_' in col and '_Count' in col]].mean()
        if models['kmeans'] is not None and len(user_features) >= min_groups * 5:
            user_groups = models['kmeans'].predict(user_features)
            original_df['PeerGroup'] = original_df['User'].map(dict(zip(user_features.index, user_groups)))
        else:
            original_df['PeerGroup'] = 0

        arima_models = {}
        for user in original_df['User'].unique():
            arima_models[user] = train_arima_model(original_df, user, 'LargeFileAccessCount')

    logger.info("Detecting anomalies with Isolation Forest...")
    y_pred = models['isolation_forest'].predict(modeling_df)
    anomaly_scores = -models['isolation_forest'].decision_function(modeling_df)
    original_df['IsAnomaly'] = (y_pred == -1)
    original_df['AnomalyScore'] = anomaly_scores

    logger.info("Detecting ARIMA anomalies...")
    original_df = detect_arima_anomalies(original_df, arima_models, 'LargeFileAccessCount')

    logger.info("Detecting peer group anomalies...")
    original_df['PeerGroupAnomaly'] = False
    original_df['PeerGroupAnomalyScore'] = 0.0
    if models['kmeans'] is not None and models['lof'] is not None:
        lof_scores = -models['lof'].score_samples(user_features)
        lof_anomalies = models['lof'].predict(user_features) == -1
        user_anomaly_map = dict(zip(user_features.index, lof_anomalies))
        user_score_map = dict(zip(user_features.index, lof_scores))
        original_df['PeerGroupAnomaly'] = original_df['User'].map(user_anomaly_map)
        original_df['PeerGroupAnomalyScore'] = original_df['User'].map(user_score_map)

    original_df['IsAnomaly'] = original_df['IsAnomaly'] | original_df['ARIMA_Anomaly'] | original_df['PeerGroupAnomaly']
    original_df['AnomalyScore'] = original_df[['AnomalyScore', 'ARIMA_AnomalyScore', 'PeerGroupAnomalyScore']].max(axis=1)
    original_df.loc[(original_df['IsAnomaly']) & (original_df['LargeFileAccessCount'] > 1), 'AnomalyScore'] *= 1.1

    logger.info(f"Total anomalies detected: {original_df['IsAnomaly'].sum()} / {len(original_df)}")

    feature_stats = {
        'Activity': original_df['Activity'].value_counts().to_dict(),
        'Hour': original_df['Hour'].value_counts().to_dict()
    }
    activity_anomalies = analyze_anomalies_by_feature(original_df, 'Activity', 'Activity', feature_stats)
    time_anomalies = analyze_anomalies_by_feature(original_df, 'Hour', 'Hour', feature_stats)
    peer_anomalies = original_df[original_df['PeerGroupAnomaly']][['User', 'PeerGroup', 'Time_Accessed', 'Filename', 'Process', 'PeerGroupAnomalyScore']].copy()
    peer_anomalies['Reason'] = "Deviation from peer group behavior"

    logger.info("\nActivity-Based Anomalies:")
    print(activity_anomalies[['User', 'Activity', 'Time_Accessed', 'AnomalyScore', 'Reason']].to_string(index=False) if not activity_anomalies.empty else "  No activity-based anomalies detected.")

    logger.info("\nTime-Based Anomalies:")
    print(time_anomalies[['User', 'Hour', 'Time_Accessed', 'AnomalyScore', 'Reason']].to_string(index=False) if not time_anomalies.empty else "  No time-based anomalies detected.")

    logger.info("\nPeer Group Anomalies:")
    print(peer_anomalies[['User', 'PeerGroup', 'Time_Accessed', 'PeerGroupAnomalyScore', 'Reason']].to_string(index=False) if not peer_anomalies.empty else "  No peer group anomalies detected.")

    if not is_identical:
        save_model_artifacts(models, encoder, scaler)
        save_history_data(combined_df, retention_days)

    logger.info("Advanced UBA retraining completed successfully.")
    return original_df, activity_anomalies, time_anomalies, peer_anomalies