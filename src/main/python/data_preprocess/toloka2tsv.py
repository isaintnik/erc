import argparse
import csv

from src.main.python.data_preprocess import toloka


MAX_SIZE = 5000000


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('path')
    args = arg_parser.parse_args()

    in_path = args.path
    out_path = in_path + '.tsv'
    data = toloka.toloka_read_raw_data(in_path, MAX_SIZE)
    data = toloka.toloka_prepare_data(data)

    with open(out_path, 'w') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for event in data:
            tsv_writer.writerow((event.uid, event.pid, event.start_ts))


if __name__ == '__main__':
    main()
