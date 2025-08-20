import unittest
import pandas as pd

############ Time-Based Format Data Utils ############
def filter_event_type(events_df, target_event_type):
    filtered_rows = {}

    for idx, row in events_df.iterrows():
        filtered_events = []

        # Count how many triplets we have
        triplet_count = len([col for col in events_df.columns if col.endswith('_date')])

        for i in range(1, triplet_count + 1):
            date = row.get(f'event_{i}_date')
            e_type = row.get(f'event_{i}_type')
            value = row.get(f'event_{i}_value')

            if isinstance(e_type, str) and target_event_type in e_type:
                filtered_events.extend([date, e_type, value])

        filtered_rows[idx] = filtered_events

    # Create new DataFrame with padded NaNs
    max_len = max((len(v) for v in filtered_rows.values()), default=0)
    filtered_df = pd.DataFrame.from_dict(filtered_rows, orient='index')

    # Rename columns to follow event_1_date, event_1_type, ...
    filtered_df.columns = [
        f'event_{i}_{suffix}'
        for i in range(1, max_len // 3 + 1)
        for suffix in ['date', 'type', 'value']
    ]

    # Drop rows where all values are NaN or None
    filtered_df = filtered_df.dropna(how='all')

    return filtered_df


def filter_event_value(events_df, target_event_value):
    filtered_rows = {}

    for idx, row in events_df.iterrows():
        filtered_events = []

        triplet_count = len([col for col in events_df.columns if col.endswith('_date')])

        for i in range(1, triplet_count + 1):
            date = row.get(f'event_{i}_date')
            e_type = row.get(f'event_{i}_type')
            value = row.get(f'event_{i}_value')

            if (
                isinstance(value, str) and
                target_event_value in value
            ):
                filtered_events.extend([date, e_type, value])

        filtered_rows[idx] = filtered_events

    # Create new DataFrame
    max_len = max((len(v) for v in filtered_rows.values()), default=0)
    filtered_df = pd.DataFrame.from_dict(filtered_rows, orient='index')

    # Rename columns
    filtered_df.columns = [
        f'event_{i}_{suffix}'
        for i in range(1, max_len // 3 + 1)
        for suffix in ['date', 'type', 'value']
    ]

    # Drop rows where all values are NaN or None
    filtered_df = filtered_df.dropna(how='all')

    return filtered_df



class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
