import pandas as pd

def remove_rare_species(df, prevalence_cutoff=0.1, avg_abundance_cutoff=0.005):
    '''

    :param df: taxon feature table normalized for relative abundance, with shape [<num of samples>, <num of taxon>]
    :param prevalence_cutoff: threshold for the percentage of samples above which the taxon is present
    :param avg_abundance_cutoff: threshold for the average percentage of samples above which the taxon is abundant
    :return:
    '''
    filt_df = df.copy()
    n_samples = df.shape[0]
    # Prevalence calculations (number of non-zero values per feature)
    frequencies = (df > 0).sum(axis=0) / n_samples
    filt_df = filt_df.loc[:, frequencies > prevalence_cutoff]
    # Average abundance calculations
    avg_abundances = df.sum(axis=0) / n_samples
    filt_df = filt_df.loc[:, avg_abundances > avg_abundance_cutoff]
    # Order
    s = filt_df.sum()
    filt_df = filt_df[s.sort_values(ascending=False).index]
    return filt_df