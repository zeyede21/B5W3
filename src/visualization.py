import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm

def plot_avg_premium_by_province(df):
    plt.figure(figsize=(12, 6))
    province_avg = df.groupby('Province')['TotalPremium'].mean().sort_values(ascending=False)
    sns.barplot(x=province_avg.index, y=province_avg.values, palette='viridis')
    plt.title('Average Insurance Premium by Province')
    plt.xlabel('Province')
    plt.ylabel('Average Premium')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_cover_type_distribution(df):
    top_provinces = df['Province'].value_counts().nlargest(5).index
    filtered = df[df['Province'].isin(top_provinces)]

    cover_province = filtered.groupby(['Province', 'CoverType']).size().unstack(fill_value=0)
    cover_province_norm = cover_province.div(cover_province.sum(axis=1), axis=0)  # Normalize

    cover_province_norm.plot(kind='bar', stacked=True, colormap='tab20', figsize=(12,7))
    plt.title('Normalized Distribution of Insurance Cover Types in Top Provinces')
    plt.xlabel('Province')
    plt.ylabel('Proportion of Cover Types')
    plt.xticks(rotation=45)
    plt.legend(title='Cover Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_premium_vs_claims_trend(df):
    plt.figure(figsize=(12, 8))
    provinces = df['Province'].unique()
    palette = sns.color_palette('tab10', n_colors=len(provinces))

    for i, province in enumerate(provinces):
        subset = df[df['Province'] == province]
        plt.scatter(subset['TotalPremium'], subset['TotalClaims'], label=province, alpha=0.6, color=palette[i])

        # Linear regression trend line
        X = subset['TotalPremium'].values
        y = subset['TotalClaims'].values
        if len(X) > 1:
            X_sm = sm.add_constant(X)
            model = sm.OLS(y, X_sm).fit()
            x_fit = np.linspace(X.min(), X.max(), 100)
            y_fit = model.predict(sm.add_constant(x_fit))
            plt.plot(x_fit, y_fit, color=palette[i], linewidth=2)

    plt.title('Total Premium vs Total Claims by Province with Trend Lines')
    plt.xlabel('Total Premium')
    plt.ylabel('Total Claims')
    plt.legend(title='Province')
    plt.tight_layout()
    plt.show()
