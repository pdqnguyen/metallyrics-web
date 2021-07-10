import pathlib

TITLE = "Vocabulary of heavy metal artists"
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
FIGURE_SIZE = (20, 10)
MARKER_SIZE = 18

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath('../data').resolve().joinpath('data.csv')