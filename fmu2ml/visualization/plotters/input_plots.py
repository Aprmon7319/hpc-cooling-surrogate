import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore
import warnings
from ..utils import get_input_column_names
warnings.filterwarnings('ignore')
# Set style
plt.style.use('seaborn-v0_8-whitegrid')

def _extract_power_columns(df, num_cdus=49):
    """Extract power columns from input dataframe"""
    power_pattern = 'Q_flow_total'
    power_cols = [col for col in df.columns if power_pattern in col]
    
    # If not found with pattern, try direct construction
    if not power_cols:
        input_cols = get_input_column_names(num_cdus=num_cdus)
        power_cols = [col for col in input_cols['Q_flow_total'] if col in df.columns]
    
    return power_cols

def plot_power_patterns(df, num_cdus=49):
    """Analyze power consumption patterns"""
    fig, axes = plt.subplots(1, 2, figsize=(25, 10))
    fig.suptitle('Power Consumption Patterns', fontsize=16)
    
    # Extract power columns
    power_cols = _extract_power_columns(df, num_cdus)
    if not power_cols:
        raise ValueError("No power columns found in dataframe")
    
    # Convert to kW if needed (assuming input is in Watts)
    power_df = df[power_cols].copy() / 1000.0
    
    # 1. Power distribution across all CDUs
    all_values = power_df.values.flatten()
    axes[0].hist(all_values, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_title('Overall Power Distribution')
    axes[0].set_xlabel('Power (kW)')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(all_values.mean(), color='red', linestyle='--', 
                     label=f'Mean: {all_values.mean():.1f} kW')
    axes[0].legend()

    # 2. Rolling standard deviation (volatility)
    time_hours = np.arange(len(power_df)) / 3600
    rolling_std = power_df.mean(axis=1).rolling(window=300).std()

    axes[1].plot(time_hours, rolling_std, 'g-', linewidth=1)
    axes[1].set_title('Power Volatility Over Time')
    axes[1].set_xlabel('Time (hours)')
    axes[1].set_ylabel('Rolling Std Dev (5-min window)')
    axes[1].grid(True, alpha=0.3)

    return fig

def plot_anomaly_detection(df):
    """Detect and visualize anomalies in CDU power data"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Anomaly Detection in CDU Power Data', fontsize=16)
    
    # 1. Z-score based anomalies
    z_scores = np.abs(zscore(df))
    anomaly_mask = z_scores > 3
    anomaly_counts = anomaly_mask.sum()
    
    axes[0, 0].bar(range(len(anomaly_counts)), anomaly_counts.values)
    axes[0, 0].set_title('Anomaly Count per CDU (|Z-score| > 3)')
    axes[0, 0].set_xlabel('CDU')
    axes[0, 0].set_ylabel('Number of Anomalies')
    
    # 2. Time distribution of anomalies
    anomaly_time = anomaly_mask.any(axis=1)
    time_hours = np.arange(len(df)) / 3600
    
    axes[0, 1].scatter(time_hours[anomaly_time], 
                      np.ones(anomaly_time.sum()) * 0.5, 
                      alpha=0.5, s=10)
    axes[0, 1].set_title('Temporal Distribution of Anomalies')
    axes[0, 1].set_xlabel('Time (hours)')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_yticks([])
    
    # 3. Power spikes detection
    rolling_mean = df.rolling(window=60).mean()  # 1-minute rolling average
    spikes = (df - rolling_mean) / rolling_mean.std()
    major_spikes = (spikes > 2).sum()
    
    axes[1, 0].bar(range(len(major_spikes)), major_spikes.values, color='red')
    axes[1, 0].set_title('Major Power Spikes per CDU')
    axes[1, 0].set_xlabel('CDU')
    axes[1, 0].set_ylabel('Number of Spikes')
    
    # 4. Anomalous CDUs visualization
    total_anomalies = anomaly_counts + major_spikes
    top_anomalous = total_anomalies.nlargest(5)
    
    for cdu in top_anomalous.index:
        axes[1, 1].plot(time_hours, df[cdu].rolling(window=180).mean(), label=cdu, alpha=0.7)

    axes[1, 1].set_title('Top 5 Most Anomalous CDUs')
    axes[1, 1].set_xlabel('Time (hours)')
    axes[1, 1].set_ylabel('Power (kW)')
    axes[1, 1].legend()
    
    plt.tight_layout()
    return fig

def plot_basic_statistics(df, num_cdus=49):
    """Plot basic statistics for all CDUs"""
    # Extract power columns
    power_cols = _extract_power_columns(df, num_cdus)
    power_df = df[power_cols].copy() / 1000.0  # Convert to kW
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('CDU Power Basic Statistics', fontsize=16)
    
    # 1. Mean power per CDU
    mean_powers = power_df.mean().sort_values(ascending=False)
    bars = axes[0, 0].bar(range(len(mean_powers)), mean_powers.values)
    axes[0, 0].set_title('Average Power per CDU')
    axes[0, 0].set_xlabel('CDU (sorted by power)')
    axes[0, 0].set_ylabel('Average Power (kW)')
    
    # Add CDU numbers on top of bars with rotation
    for i, (bar, cdu_name) in enumerate(zip(bars, mean_powers.index)):
        height = bar.get_height()
        cdu_num = cdu_name.split('_')[4] if '_' in cdu_name else str(i+1)
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{cdu_num}', ha='center', va='bottom', 
                       rotation=90, fontsize=7)
    
    # 2. Standard deviation per CDU
    std_powers = power_df.std().sort_values(ascending=False)
    bars = axes[0, 1].bar(range(len(std_powers)), std_powers.values, color='orange')
    axes[0, 1].set_title('Power Variability (Std Dev) per CDU')
    axes[0, 1].set_xlabel('CDU (sorted by variability)')
    axes[0, 1].set_ylabel('Standard Deviation (kW)')
    
    for i, (bar, cdu_name) in enumerate(zip(bars, std_powers.index)):
        height = bar.get_height()
        cdu_num = cdu_name.split('_')[4] if '_' in cdu_name else str(i+1)
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{cdu_num}', ha='center', va='bottom',
                       rotation=90, fontsize=7)
    
    # 3. Box plot of power ranges
    sample_cdus = power_df.columns[::5]  # Every 5th CDU
    axes[1, 0].boxplot([power_df[cdu].values for cdu in sample_cdus], 
                       labels=[col.split('_')[4] if '_' in col else str(i) 
                              for i, col in enumerate(sample_cdus)])
    axes[1, 0].set_title('Power Distribution (Sample CDUs)')
    axes[1, 0].set_xlabel('CDU')
    axes[1, 0].set_ylabel('Power (kW)')
    plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=60)
    
    # 4. Coefficient of variation
    cv = (power_df.std() / power_df.mean()).sort_values(ascending=False)
    bars = axes[1, 1].bar(range(len(cv)), cv.values, color='green')
    axes[1, 1].set_title('Coefficient of Variation per CDU')
    axes[1, 1].set_xlabel('CDU (sorted by CV)')
    axes[1, 1].set_ylabel('CV (std/mean)')
    
    for i, (bar, cdu_name) in enumerate(zip(bars, cv.index)):
        height = bar.get_height()
        cdu_num = cdu_name.split('_')[4] if '_' in cdu_name else str(i+1)
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{cdu_num}', ha='center', va='bottom',
                       rotation=90, fontsize=7)
    
    plt.tight_layout()
    return fig

def plot_time_series_overview_with_scenarios(data, df_scenario):
    """Plot time series overview with aggregated metrics grouped by scenario type"""
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig.suptitle('Time Series Overview by Scenario Type - 8 Hour Period', fontsize=16)
    
    # Convert data dictionary to DataFrame if needed
    if isinstance(data, dict):
        data = pd.DataFrame(data)
    
    # Convert scenario dictionary to DataFrame
    df_scenario = pd.DataFrame(df_scenario)
    
    # Apply rolling mean
    df = data.rolling(window=30).mean()
    
    # Create time index in hours
    time_hours = np.arange(len(df)) / 3600
    
    # Define colors for each scenario type
    scenario_colors = {
        'normal': 'green',
        'edge': 'orange', 
        'fault': 'red'
    }
    
    # Initialize arrays for each scenario type
    scenario_data = {scenario: {
        'total_power': np.zeros(len(df)),
        'sum_power': np.zeros(len(df)),
        'count': np.zeros(len(df)),
        'min_power': np.full(len(df), np.inf),
        'max_power': np.full(len(df), -np.inf),
        'active_count': np.zeros(len(df))
    } for scenario in scenario_colors}
    
    # Calculate idle threshold
    idle_threshold = df.mean().mean() * 0.5
    
    # For each time point in the rolled data
    for t in range(len(df)):
        # Map rolled time index back to original scenario time index
        # Since we're using a rolling window, we need to handle the mapping carefully
        scenario_time_idx = min(t // 3600, len(df_scenario) - 1)  # Assuming hourly scenario changes
        
        # Process each CDU
        for cdu in df.columns:
            if cdu in df_scenario.columns:
                # Get the scenario for this CDU at this time
                scenario = df_scenario.loc[scenario_time_idx, cdu]
                
                # Get power value
                power = df.loc[t, cdu]
                
                if not pd.isna(power) and scenario in scenario_colors:
                    # Update totals
                    scenario_data[scenario]['total_power'][t] += power
                    scenario_data[scenario]['sum_power'][t] += power
                    scenario_data[scenario]['count'][t] += 1
                    
                    # Update min/max
                    scenario_data[scenario]['min_power'][t] = min(
                        scenario_data[scenario]['min_power'][t], power)
                    scenario_data[scenario]['max_power'][t] = max(
                        scenario_data[scenario]['max_power'][t], power)
                    
                    # Count active CDUs
                    if power > idle_threshold:
                        scenario_data[scenario]['active_count'][t] += 1
    
    # Calculate averages and handle empty scenarios
    for scenario in scenario_colors:
        mask = scenario_data[scenario]['count'] > 0
        avg_power = np.full(len(df), np.nan)
        avg_power[mask] = (scenario_data[scenario]['sum_power'][mask] / 
                          scenario_data[scenario]['count'][mask])
        scenario_data[scenario]['avg_power'] = avg_power
        
        # Set inf values to nan for plotting
        scenario_data[scenario]['min_power'][scenario_data[scenario]['min_power'] == np.inf] = np.nan
        scenario_data[scenario]['max_power'][scenario_data[scenario]['max_power'] == -np.inf] = np.nan
    
    # 1. Total data center power by scenario
    for scenario in ['normal', 'edge', 'fault']:
        axes[0].plot(time_hours, scenario_data[scenario]['total_power'], 
                    color=scenario_colors[scenario], linewidth=2, label=scenario.capitalize())
    axes[0].set_title('Total Power by Scenario Type')
    axes[0].set_ylabel('Total Power (kW)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Average CDU power with min-max range by scenario
    for scenario in ['normal', 'edge', 'fault']:
        # Plot average
        axes[1].plot(time_hours, scenario_data[scenario]['avg_power'], 
                    color=scenario_colors[scenario], linewidth=2, label=f'{scenario.capitalize()} avg')
        
        # Fill between min and max with transparency
        valid_mask = ~np.isnan(scenario_data[scenario]['min_power'])
        if np.any(valid_mask):
            axes[1].fill_between(time_hours[valid_mask], 
                                scenario_data[scenario]['min_power'][valid_mask], 
                                scenario_data[scenario]['max_power'][valid_mask], 
                                alpha=0.2, color=scenario_colors[scenario])
    
    axes[1].set_title('CDU Power Statistics by Scenario Type')
    axes[1].set_ylabel('Power (kW)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Number of active CDUs by scenario
    for scenario in ['normal', 'edge', 'fault']:
        axes[2].plot(time_hours, scenario_data[scenario]['active_count'], 
                    color=scenario_colors[scenario], linewidth=2, label=scenario.capitalize())
    axes[2].set_title('Number of Active CDUs by Scenario Type')
    axes[2].set_xlabel('Time (hours)')
    axes[2].set_ylabel('Count')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
def plot_time_series_overview(data, num_cdus=49):
    """Plot time series overview with aggregated metrics"""
    # Extract power columns
    power_cols = _extract_power_columns(data, num_cdus)
    df = data[power_cols].copy() / 1000.0  # Convert to kW
    df = df.rolling(window=60).mean()
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig.suptitle('Time Series Overview', fontsize=16)
    
    # Create time index in hours
    time_hours = np.arange(len(df)) / 3600
    
    # 1. Total data center power
    total_power = df.sum(axis=1)
    axes[0].plot(time_hours, total_power, 'b-', linewidth=2)
    axes[0].set_title('Total Data Center Power')
    axes[0].set_ylabel('Total Power (kW)')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Average, min, max CDU power
    axes[1].plot(time_hours, df.mean(axis=1), 'g-', linewidth=2, label='Average')
    axes[1].fill_between(time_hours, df.min(axis=1), df.max(axis=1), 
                        alpha=0.3, color='gray', label='Min-Max Range')
    axes[1].set_title('CDU Power Statistics Over Time')
    axes[1].set_ylabel('Power (kW)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Number of active CDUs (above idle threshold)
    idle_threshold = df.mean().mean() * 0.5
    active_cdus = (df > idle_threshold).sum(axis=1)
    axes[2].plot(time_hours, active_cdus, 'r-', linewidth=2)
    axes[2].set_title('Number of Active CDUs')
    axes[2].set_xlabel('Time (hours)')
    axes[2].set_ylabel('Count')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_cdu_clustering(df):
    """Cluster CDUs based on power patterns"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('CDU Clustering Analysis', fontsize=16)
    
    # Calculate features for clustering
    features = pd.DataFrame({
        'mean': df.mean(),
        'std': df.std(),
        'max': df.max(),
        'min': df.min(),
        'range': df.max() - df.min()
    })
    
    # 1. Scatter plot: Mean vs Std
    axes[0, 0].scatter(features['mean'], features['std'], s=50, alpha=0.6)
    for idx, cdu in enumerate(features.index):
        if idx % 10 == 0:  # Label every 10th CDU
            axes[0, 0].annotate(cdu, (features.loc[cdu, 'mean'], features.loc[cdu, 'std']))
    axes[0, 0].set_xlabel('Mean Power (kW)')
    axes[0, 0].set_ylabel('Std Dev (kW)')
    axes[0, 0].set_title('CDU Power Characteristics')
    
    # 2. Scatter plot: Mean vs Range
    axes[0, 1].scatter(features['mean'], features['range'], s=50, alpha=0.6, c='orange')
    axes[0, 1].set_xlabel('Mean Power (kW)')
    axes[0, 1].set_ylabel('Power Range (kW)')
    axes[0, 1].set_title('Mean vs Range')
    
    # 3. Hierarchical clustering dendrogram
    from scipy.cluster.hierarchy import dendrogram, linkage
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    linkage_matrix = linkage(features_scaled, method='ward')
    
    dendrogram(linkage_matrix, labels=features.index, ax=axes[1, 0])
    axes[1, 0].set_title('CDU Hierarchical Clustering')
    axes[1, 0].set_xlabel('CDU')
    axes[1, 0].set_ylabel('Distance')
    plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=90)
    
    # 4. K-means clustering (3 clusters)
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(features_scaled)
    
    scatter = axes[1, 1].scatter(features['mean'], features['std'], 
                                c=clusters, cmap='viridis', s=50, alpha=0.6)
    axes[1, 1].set_xlabel('Mean Power (kW)')
    axes[1, 1].set_ylabel('Std Dev (kW)')
    axes[1, 1].set_title('K-means Clustering (3 clusters)')
    plt.colorbar(scatter, ax=axes[1, 1], label='Cluster')
    
    plt.tight_layout()
    return fig


def plot_cdu_input_power_minimal(df: pd.DataFrame, 
                          compute_blocks: list = [13, 16, 18, 38, 42],
                          figsize: tuple = (12, 6),
                          save_path: str = None, rolling_window: int = 180, range_limit: tuple = (0, 3600)):
    """
    Minimal visualization of CDU input power for specific compute blocks
    
    Parameters:
    - df: DataFrame with FMU output data
    - compute_blocks: List of compute block numbers to visualize
    - figsize: Figure size (width, height)
    - save_path: Optional path to save the figure
    """
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    
    # Plot each CDU
    for idx, cb in enumerate(compute_blocks):
        col_name = f'CDU_{0 if cb < 10 else ""}{cb}'
        
        if col_name in df.columns:
            power_diff = df.loc[range_limit[0]:range_limit[1], col_name].rolling(window=rolling_window).mean()

            # Plot with distinct color and style
            ax.plot(power_diff.index, power_diff, 
                   label=f'CDU {cb}', 
                   color=colors[idx % len(colors)],
                   linewidth=2,
                   alpha=0.8)
    
    # Styling
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Power (kW)', fontsize=12)
    ax.set_title('CDU Input Power Comparison', fontsize=14, pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return fig, ax

def plot_cdu_input_power(df: pd.DataFrame, 
                         compute_blocks: list = None,
                         figsize: tuple = (12, 6),
                         rolling_window: int = 180,
                         num_cdus: int = 49):
    """Plot CDU input power for specific compute blocks"""
    
    if compute_blocks is None:
        compute_blocks = [13, 16, 17, 38, 42]
    
    # Get input column names
    input_cols = get_input_column_names(num_cdus=num_cdus)
    
    fig, ax = plt.subplots(figsize=figsize)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, cb in enumerate(compute_blocks):
        col_name = f'simulator_1_datacenter_1_computeBlock_{cb}_cabinet_1_sources_Q_flow_total'
        
        if col_name in df.columns:
            # Convert to kW
            power_kw = df[col_name].values / 1000.0
            power_smooth = pd.Series(power_kw).rolling(window=rolling_window, center=True).mean()
            
            ax.plot(power_smooth.index, power_smooth.values, 
                   label=f'CDU {cb}', color=colors[idx % len(colors)], linewidth=2)
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Power (kW)', fontsize=12)
    ax.set_title('CDU Input Power', fontsize=14, pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    return fig, ax