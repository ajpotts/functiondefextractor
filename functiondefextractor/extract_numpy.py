import sys

from condition_checker import check_condition
from core_extractor import extractor
from core_extractor import get_report
from extractor_cmd import validate_inputs
from extractor_cmd import create_parser

import pandas as pd

if __name__ == '__main__':

    scipy_path = "/home/amandapotts/git/scipy/scipy"
    print(scipy_path)

    #   Extract the function calls, and enhance with extracted function names.
    df = extractor(scipy_path)

    df['np_calls'] = df['Code'].str.findall(r"np\.([\w\.]*)")
    df['num_np_calls'] = df['np_calls'].apply(lambda x: len(x))
    df['np_calls_w_args'] = df['Code'].str.findall(r"np\.([\w_\.]+\(.*\))")
    df['scipy_function'] = df['Uniq ID'].str.findall(r"/home/amandapotts/git/scipy/scipy/([\w_/]*)")
    df = df.explode('scipy_function')

    my_out = "/home/amandapotts/git/functiondefextractor/data/scipy/scipy_function_definitions.csv"
    df.to_csv(my_out)

    # Compute the number of numpy calls, over all, and by number of scipy functions.
    df_np_calls = df.explode('np_calls')[["scipy_function","np_calls"]]
    df_np_calls = df_np_calls[df_np_calls['np_calls'].notnull()]
    df_np_calls = df_np_calls.explode("np_calls")

    total_np_calls = df_np_calls.groupby("np_calls").size()
    df_total_np_calls = pd.DataFrame({"total_np_calls": total_np_calls})

    num_scipy_funs_used = df_np_calls.drop_duplicates().groupby("np_calls").size()
    df_num_scipy_funs_used = pd.DataFrame({"num_scipy_funs_used_by":num_scipy_funs_used})

    df_call_stats = df_num_scipy_funs_used.join(df_total_np_calls, lsuffix='_caller', rsuffix='_other')
    df_call_stats = df_call_stats.sort_values(by=['num_scipy_funs_used_by', 'total_np_calls'], ascending=False)

    my_out2 = "/home/amandapotts/git/functiondefextractor/data/scipy/np_call_stats.csv"
    df_call_stats.to_csv(my_out2)


    df_args = df.explode('np_calls_w_args')
    df_args = df_args[df_args['np_calls_w_args'].notnull()][["scipy_function", "np_calls_w_args"]]
    df_args['np_call'] = df_args['np_calls_w_args'].str.findall(r"(^[\w\.]*)")
    df_args = df_args.explode('np_call')
    df_args['np_args'] = df_args['np_calls_w_args'].str.findall(r"([\w\s\._]*)=")
    df_args = df_args.explode('np_args')
    df_args = df_args[df_args['np_args'].notnull()]
    df_args = df_args[["scipy_function", "np_call", "np_args"]]

    total_np_calls = df_args.groupby(["np_call", "np_args"]).size()
    df_total_np_calls = pd.DataFrame({"total_np_calls": total_np_calls})

    num_scipy_funs_used = df_args.drop_duplicates().groupby(["np_call", "np_args"]).size()
    df_num_scipy_funs_used = pd.DataFrame({"num_scipy_funs_used_by": num_scipy_funs_used})

    df_call_stats = df_num_scipy_funs_used.join(df_total_np_calls, lsuffix='_caller', rsuffix='_other')
    df_call_stats = df_call_stats.sort_values(by=['num_scipy_funs_used_by', 'total_np_calls'], ascending=False)


    my_out3 = "/home/amandapotts/git/functiondefextractor/data/scipy/np_arg_stats.csv"
    df_call_stats.to_csv(my_out2)