import sys

from condition_checker import check_condition
from core_extractor import extractor
from core_extractor import get_report
from extractor_cmd import validate_inputs
from extractor_cmd import create_parser

if __name__ == '__main__':

    scipy_path = "/home/amandapotts/git/scipy/scipy"
    print(scipy_path)

    df = extractor(scipy_path)

    df['np_calls'] = df['Code'].str.findall(r"np\.([\w\.]*)")
    df['num_np_calls'] = df['np_calls'].apply(lambda x: len(x))
    df['function'] = df['Uniq ID'].str.findall(r"/home/amandapotts/git/scipy/scipy/([\w_/]*)")


    my_out = "/home/amandapotts/git/functiondefextractor/data/scipy/scipy_function_definitions.csv"

    print(df)

    df.to_csv(my_out)
