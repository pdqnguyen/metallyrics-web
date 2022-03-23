import pathlib

# App metadata
TITLE = "Vocabulary of heavy metal artists"

# Feature set
NUM_WORDS = 10000
NUM_BANDS = 200
FEATURES = {
    'reviews': 'Number of album reviews',
    'rating': 'Average album review score',
    'unique_first_words': f"Number of unique words in first {NUM_WORDS:,.0f} words",
    'word_count': 'Total number of words in discography',
    'words_per_song': 'Average words per song',
    'words_per_song_uniq': 'Average unique words per song',
    'seconds_per_song': 'Average song length in seconds',
    'word_rate': 'Average words per second',
    'word_rate_uniq': f"Average unique words per second",
    'types': 'Types (total unique words)',
    'TTR': 'Type-token ratio (TTR)',
    'logTTR': 'Log-corrected TTR',
    'MTLD': 'MTLD',
    'logMTLD': 'Log(MTLD)',
    'vocd-D': 'vocd-D',
    'logvocd-D': 'Log(vocd-D)',
}

# Seaborn swarm plot parameters
FIGURE_SIZE = (20, 8)
MARKER_SIZE = 13

# Plotly scatter parameters
PLOT_KWARGS = {
    'autosize': False,
    'showlegend': False,
    'hoverlabel': dict(bgcolor='#730000', font_color='#EBEBEB', font_family='Monospace'),
    'template': 'plotly_dark',
}
AXES_KWARGS = {
    'gridwidth': 2,
    'gridcolor': '#444444',
}

# Dataset location
PATH = pathlib.Path(__file__).parent
LYRICAL_COMPLEXITY_PATH = PATH.joinpath('../data').resolve().joinpath('bands-lyrical-complexity.csv')
WORDCLOUD_PATH = PATH.joinpath('../data').resolve().joinpath('bands-wordclouds.csv')