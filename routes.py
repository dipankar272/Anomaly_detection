from flask import Blueprint, render_template, jsonify, request
from .models import periodic_retrain
import pandas as pd
import os

bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        # Handle file upload
        if 'csv_file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['csv_file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'File must be a CSV'}), 400

        # Save the file temporarily
        temp_path = 'temp_file.csv'
        file.save(temp_path)

        # Run analysis
        original_df, activity_anomalies, time_anomalies, peer_anomalies = periodic_retrain(temp_path)

        # Clean up
        os.remove(temp_path)

        # Prepare data for Activity Timeline
        timeline_data = {
            'dates': original_df['Time_Accessed'].dt.strftime('%Y-%m-%d').unique().tolist(),
            'normal_events': [],
            'anomalous_events': []
        }
        for date in timeline_data['dates']:
            date_df = original_df[original_df['Time_Accessed'].dt.strftime('%Y-%m-%d') == date]
            timeline_data['normal_events'].append(len(date_df[~date_df['IsAnomaly']]))
            timeline_data['anomalous_events'].append(len(date_df[date_df['IsAnomaly']]))

        # Prepare data for Anomaly Types (Donut Chart)
        anomaly_types = activity_anomalies['Activity'].value_counts().to_dict()

        # Prepare data for Activity Distribution
        activity_dist = original_df['Activity'].value_counts().to_dict()

        # Convert anomaly dataframes to list of dicts for JSON serialization
        activity_anomalies_data = activity_anomalies[['User', 'Activity', 'Time_Accessed', 'AnomalyScore', 'Reason']].to_dict('records')
        time_anomalies_data = time_anomalies[['User', 'Hour', 'Time_Accessed', 'AnomalyScore', 'Reason']].to_dict('records')
        peer_anomalies_data = peer_anomalies[['User', 'PeerGroup', 'Time_Accessed', 'PeerGroupAnomalyScore', 'Reason']].to_dict('records')

        return jsonify({
            'timeline': timeline_data,
            'anomaly_types': anomaly_types,
            'activity_distribution': activity_dist,
            'total_activity': len(original_df),
            'total_anomalies': int(original_df['IsAnomaly'].sum()),
            'activity_anomalies': activity_anomalies_data,
            'time_anomalies': time_anomalies_data,
            'peer_anomalies': peer_anomalies_data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500