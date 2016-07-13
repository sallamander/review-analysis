"""A script for general eda."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_hist_kde(data, bins=20, hist=True, kde=True, title=None, ylabel=None,
                  xlabel=None, xlim=None, xticks=None, save_fp=None): 
    """Plot a histogram and/or kde of the inputted data.

    Args:
    ----
        data: 1p nd.ndarray
        bins (optional): int
        hist (optional): bool
        kde (optional): bool
        title (optional): str
        ylabel (optional): str
        xlabel (optional): str
        xlim (optional): tuple of ints
        xticks (optional): range object
        save_fp (optional): str
    """

    sns.distplot(data, bins=bins, hist=hist, kde=kde)

    if title:
        plt.title(title)
    if ylabel: 
        plt.ylabel(ylabel)
    if xlabel: 
        plt.xlabel(xlabel)
    if xlim:
        plt.xlim(xlim)
    if xticks: 
        plt.xticks(xticks)
    plt.grid(False)

    if save_fp: 
        plt.savefig(save_fp)
        plt.close()
    else: 
        plt.show()

def plot_boxplot(ratios, xlim=None, title=None, xlabel=None, save_fp=None):
    """Plot a boxplot of the inputted data.

    Args:
    ----
        ratios: 1d np.ndarray
        save_fp (optional): str
    """

    sns.boxplot(ratios)

    if title:
        plt.title(title)
    if xlabel: 
        plt.xlabel(xlabel)
    plt.grid(False)

    if xlim:
        plt.xlim(xlim)
    if save_fp: 
        plt.savefig(save_fp)
        plt.close()
    else: 
        plt.show()

if __name__ == '__main__':
    filtered_reviews_fp = 'work/reviews/amazon/raw_food_reviews.csv'
    filtered_reviews_df = pd.read_csv(filtered_reviews_fp)

    helpfulness_denom = filtered_reviews_df['helpfulness_denominator']
    xlim = (0, 50)
    ylabel = 'Frequency' 
    xlabel = 'Number of Votes'
    title = 'Frequency of Helpfulness Votes for Amazon food reviews' 

    plot_hist_kde(helpfulness_denom, bins=500, kde=False, title=title, 
                  ylabel=ylabel, xlabel=xlabel, xlim=xlim, xticks=range(0, 50, 5), 
                  save_fp='work/temp_viz/votes_xlim_0_50.png')

    xlim = (0, 20)
    plot_hist_kde(helpfulness_denom, bins=500, kde=False, title=title, 
                  ylabel=ylabel, xlabel=xlabel, xlim=xlim, xticks=range(0, 20, 2), 
                  save_fp='work/temp_viz/votes_xlim_0_20.png')

    query_str = 'helpfulness_denominator > @vote_frequency'
    title = 'Distribution of Helpfulness Ratio'
    xlim = (0, 1)
    for vote_frequency in range(0, 55, 5): 
        num_obs = (helpfulness_denom > vote_frequency).sum()
        filtered_by_votes = filtered_reviews_df.query(query_str)
        ratios = filtered_by_votes['helpfulness_ratio']
        print('-' * 50)
        print('Vote frequency minimum: {}'.format(vote_frequency))
        print('Number of obs: {}'.format(num_obs))
        print('Ratio Mean: {}'.format(ratios.mean()))
        print('Ratio Variance: {}'.format(ratios.var()))

        xlabel = 'Helpfulness Ratio - Greater than {} Votes'.format(vote_frequency)
        save_fp='work/temp_viz/ratio_boxplot_{}_votes.png'.format(vote_frequency)
        plot_boxplot(ratios, xlim=xlim, title=title, xlabel=xlabel, save_fp=save_fp)
        save_fp='work/temp_viz/ratio_hist_{}_votes.png'.format(vote_frequency)
        plot_hist_kde(ratios, bins=50, kde=False, title=title, 
                ylabel=ylabel, xlabel=xlabel, save_fp=save_fp, xlim=xlim)
