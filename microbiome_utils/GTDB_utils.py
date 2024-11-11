import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def map_taxon_level(df, column_name='taxon'):
    '''

    :param df: feature_table_gtdb.tsv output file from GTDB R script -
    df with taxon classification and sample ids read counts as columns
    :param column_name: name of the taxon column
    :return: dictionary of taxon levels (lowest classification resolution for each classification)
    (for example: {'d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__;f__;g__': 'c'})
    '''
    taxon_mapping = {}

    # Loop through each taxon in the DataFrame
    for taxon in df[column_name]:
        if taxon=='d__;p__;c__;o__;f__;g__':
            taxon_mapping[taxon]='u'
        # Split the taxon string by ';' to get individual levels
        levels = taxon.split(';')

        # Reverse the levels and find the last non-empty level
        for level in reversed(levels):
            # Check if level has a valid taxonomic identifier (not '__')
            if '__' in level and len(level.split('__')[-1].strip()) > 0:
                # Take the first character before the '__' and map it
                taxon_mapping[taxon] = level[0]
                break

    return taxon_mapping


def group_reads_by_taxa_level(df, taxon_mapping, taxon_column='taxon'):
    '''

    :param df: feature_table_gtdb.tsv output file from GTDB R script -
    df with taxon classification and sample ids read counts as columns
    :param taxon_mapping: dict of taxon levels (output of map_taxon_level)
    :param taxon_column: name of the taxon column in df
    :return: df with taxa level as index ('c'/'d'/'f' etc.), and read counts of each sample id as columns.
    '''
    # Step 1: Add a new column 'taxa_level' by mapping the 'taxon' column using the dictionary
    df['taxa_level'] = df[taxon_column].map(taxon_mapping)

    # Step 2: Group by 'taxa_level' and sum the read counts for each sample ID column
    # Exclude 'taxon' and 'taxa_level' from the sum operation (assuming they are not read counts)
    read_count_columns = df.columns.difference([taxon_column, 'taxa_level'])

    # Group by 'taxa_level' and sum the read counts for each sample
    grouped_df = df.groupby('taxa_level')[read_count_columns].sum()

    # Step 3: Return the new DataFrame with 'taxa_level' as the index and sample IDs as columns
    return grouped_df

def normalize_columns_by_sum(reads_by_taxa_level_df):
    '''

    :param reads_by_taxa_level_df: output of group_reads_by_taxa_level
    :return: normalized reads_by_taxa_level_df per column (divided by column sum)
    '''
    # Normalize each column by its sum
    reads_pcntg_by_taxa_level_df = reads_by_taxa_level_df.div(reads_by_taxa_level_df.sum(axis=0), axis=1)
    return reads_pcntg_by_taxa_level_df


def generate_plot_from_reads_pcntg_by_taxa_level_df(df, plot_img_path_dir='.'):
    '''
    :param df: output of normalize_columns_by_sum (reads_pcntg_by_taxa_level_df)
    :param plot_img_path_dir:
    :return: plots and saves as png file the Percent of reads assigned to each taxonomic rank,
    ordered by taxa level, and returns the melted df used for plot for debug
    '''
    # Mapping taxa levels to their full names
    taxa_mapping = {'d': '1-Domain', 'p': '2-Phylum', 'c': '3-Class', 'o': '4-Order', 'f': '5-Family', 'g': '6-Genus',
                    'u': '7-Unassigned'}

    # Desired order for the taxonomic ranks
    taxa_order = ['1-Domain', '2-Phylum', '3-Class', '4-Order', '5-Family', '6-Genus', '7-Unassigned']

    # Melt the DataFrame for seaborn's input
    df_melt = df.reset_index().melt(id_vars=['taxa_level'], var_name='sample_id', value_name='percent_reads')

    # Map the 'taxa_level' column and ensure the correct order
    df_melt['taxa_level'] = df_melt['taxa_level'].map(taxa_mapping)
    df_melt['taxa_level'] = pd.Categorical(df_melt['taxa_level'], categories=taxa_order, ordered=True)

    # Plot the data
    plt.figure(figsize=(10, 6))
    sns.stripplot(x='taxa_level', y='percent_reads', data=df_melt, jitter=True, color='black', alpha=0.5)

    # Add a line for the median of each taxonomic rank
    sns.pointplot(x='taxa_level', y='percent_reads', data=df_melt, estimator='median', join=False, color='black')

    # Customize the plot
    plt.title('Percent of reads assigned to each taxonomic rank')
    plt.ylabel('Percent of reads')
    plt.xlabel('Taxonomic rank')
    plt.xticks(rotation=45)

    # Show the plot
    plt.tight_layout()
    plt.savefig(f'{plot_img_path_dir}/Percent_of_reads_assigned_to_each_taxonomic_rank.png')
    plt.show()

    return df_melt

def plot_reads_pcnt_by_taxon_rank(feature_table_gtdb: pd.DataFrame, plot_img_path_dir='.'):
    '''
    plots Percent of reads assigned to each taxonomic rank and saves to png file
    :param feature_table_gtdb: feature_table_gtdb.tsv output file from GTDB R script -
    df with taxon classification and sample ids read counts as columns
    :return: melted df used for plot for debug
    '''
    taxon_mapping = map_taxon_level(feature_table_gtdb)
    reads_by_taxa_level_df = group_reads_by_taxa_level(feature_table_gtdb, taxon_mapping)
    reads_pcntg_by_taxa_level_df = normalize_columns_by_sum(reads_by_taxa_level_df)
    df_melt = generate_plot_from_reads_pcntg_by_taxa_level_df(reads_pcntg_by_taxa_level_df, plot_img_path_dir)

    return df_melt


def filter_rows_by_taxon(feature_table_gtdb, taxon_mapping, taxon_level='g'):
    '''

    :param feature_table_gtdb: feature_table_gtdb.tsv output file from GTDB R script -
    df with taxon classification and sample ids read counts as columns
    :param taxon_mapping: dict of taxon levels (output of map_taxon_level)
    :param taxon_level: char representing taxon level
    taxa mapping: {'d': '1-Domain', 'p': '2-Phylum', 'c': '3-Class',
    'o': '4-Order', 'f': '5-Family', 'g': '6-Genus', 'u': '7-Unassigned'}
    :return: filtered df of rows with taxon classification in the specified resolution
    '''
    # Create a boolean mask for rows where the taxon maps to 'g'
    df = feature_table_gtdb.copy()
    mask = df['taxon'].map(taxon_mapping) == taxon_level

    # Filter the DataFrame using the mask
    filtered_taxon_df = df[mask]
    assert filtered_taxon_df.taxa_level.unique()[0] == taxon_level

    return filtered_taxon_df

def normalize_per_column_to_relative_abundance(feature_table_df: pd.DataFrame,):
    '''

    :param feature_table_df: feature table with shape (<#samples>, <#taxon>)
    :return:
    '''
    norm_df = feature_table_df.div(feature_table_df.sum(axis=0), axis=1)
    return norm_df

def create_taxa_filtered_normalized_df(feature_table_gtdb: pd.DataFrame, taxon_mapping: dict[str, str],
                                        taxon_level='g', remove_taxa_level_col: bool = True):
    '''

    :param feature_table_gtdb: feature_table_gtdb.tsv output file from GTDB R script -
    df with taxon classification and sample ids read counts as columns,
    :param taxon_mapping: dict of taxon levels (output of map_taxon_level)
    :param taxon_level: char representing taxon level
    taxa mapping: {'d': '1-Domain', 'p': '2-Phylum', 'c': '3-Class',
    'o': '4-Order', 'f': '5-Family', 'g': '6-Genus', 'u': '7-Unassigned'}
    :param remove_taxa_level_col: remove taxon level column if present from previous functions
    :return: re-normalized df filtered by taxon level, with taxon as index and samples as columns
    '''
    genus_assigned_df = filter_rows_by_taxon(feature_table_gtdb, taxon_mapping, taxon_level)
    if remove_taxa_level_col:
        genus_assigned_df = genus_assigned_df.drop(columns='taxa_level')
    genus_assigned_df = genus_assigned_df.set_index('taxon')
    # Normalize to relative abundance
    norm_genus_df = genus_assigned_df.div(genus_assigned_df.sum(axis=0), axis=1)
    norm_genus_df = norm_genus_df.T
    return norm_genus_df
