import sys

from condition_checker import check_condition
from core_extractor import extractor
from core_extractor import get_report
from extractor_cmd import validate_inputs
from extractor_cmd import create_parser

if __name__ == '__main__':

    my_path = "/home/amandapotts/git/scipy/scipy"
    print(my_path)

    df = extractor(my_path)
    print(df)

    #df['numpy'] = df['Code'].str.extractall(r"np.(\w*)")

    my_out = "/home/amandapotts/git/personal-dev/data.csv"

    df.to_csv(my_out)

    report = get_report(df, my_path)
    print(report)
    print(type(report))