# digital_twin/data_analyzer.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path='simulation_results.csv'):
    df = pd.read_csv(file_path)
    return df

def explore_final_status_distribution(df):
    plt.figure(figsize=(8, 6))
    sns.countplot(x='final_status', data=df)
    plt.title("Final Production Status Distribution")
    plt.xlabel("Final Status")
    plt.ylabel("Count")
    plt.savefig("final_status_distribution.png")
    # plt.show()

def correlation_analysis(df):
    # Select only numeric columns for correlation analysis
    numeric_cols = ['mixing_time', 'mixing_speed', 'uniformity_index', 'granulation_time',
                    'binder_rate', 'granule_density', 'drying_temp', 'moisture_content',
                    'comp_pressure', 'tablet_hardness', 'weight_variation', 'dissolution', 'yield_percent']
    corr_matrix = df[numeric_cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix of Process Parameters")
    plt.show()

def main():
    df = load_data()
    explore_final_status_distribution(df)
    correlation_analysis(df)
    
if __name__ == "__main__":
    main()
