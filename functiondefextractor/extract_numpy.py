import sys

from condition_checker import check_condition
from core_extractor import extractor
from core_extractor import get_report
from extractor_cmd import validate_inputs
from extractor_cmd import create_parser

import pandas as pd


def add_numpy_calls(df):
    df["np_calls"] = df["Code"].str.findall(r"np\.([\w\._]*)")
    df["num_np_calls"] = df["np_calls"].apply(lambda x: len(x))
    df["np_calls_w_args"] = df["Code"].str.findall(r"np\.([\w\_\.]+\(.*\))")

    return df


def add_library_functions(df: pd.DataFrame, lib_name: str, lib_path: str):
    df[lib_name + "_function"] = df["Uniq ID"].str.findall(
        #r"/home/amandapotts/git/scipy/scipy/([\w_/]*)"
        lib_path+"([\w_/]*)"
    )
    df = df.explode(lib_name + "_function")
    return df

def get_and_write_np_stats(lib_name:str,lib_path:str, out_dir:str):
    df = extractor(lib_path)
    df = add_numpy_calls(df)
    df = add_library_functions(df, lib_name, lib_path)
    my_out = out_dir + lib_name + "_function_definitions.csv"
    df.to_csv(my_out)
    return df

if __name__ == "__main__":
    out_dir = "/home/amandapotts/git/functiondefextractor/data/out/"
    scipy_path = "/home/amandapotts/git/scipy/scipy"
    pandas_path =  "/home/amandapotts/git/pandas/pandas"

    #   Extract the function calls, and enhance with extracted function names.
    df = get_and_write_np_stats("scipy", scipy_path, out_dir)

    # Compute the number of numpy calls, over all, and by number of scipy functions.
    df_np_calls = df.explode("np_calls")[["scipy_function", "np_calls"]]
    df_np_calls = df_np_calls[df_np_calls["np_calls"].notnull()]
    df_np_calls = df_np_calls.explode("np_calls")

    total_np_calls = df_np_calls.groupby("np_calls").size()
    df_total_np_calls = pd.DataFrame({"total_np_calls": total_np_calls})

    num_scipy_funs_used = df_np_calls.drop_duplicates().groupby("np_calls").size()
    df_num_scipy_funs_used = pd.DataFrame({"num_scipy_funs_used_by": num_scipy_funs_used})

    df_call_stats = df_num_scipy_funs_used.join(df_total_np_calls, lsuffix="_caller", rsuffix="_other")
    df_call_stats = df_call_stats.sort_values(
        by=["num_scipy_funs_used_by", "total_np_calls"], ascending=False
    )

    my_out2 = out_dir + "scipy_np_call_stats.csv"
    df_call_stats.to_csv(my_out2)

    df_args = df.explode("np_calls_w_args")
    df_args = df_args[df_args["np_calls_w_args"].notnull()][["scipy_function", "np_calls_w_args"]]
    df_args["np_call"] = df_args["np_calls_w_args"].str.findall(r"(^[\w\.]*)")
    df_args = df_args.explode("np_call")
    df_args["np_args"] = df_args["np_calls_w_args"].str.findall(r"([\w\._]*)\s*=[^=<>!]")
    df_args = df_args.explode("np_args")
    df_args = df_args[df_args["np_args"].notnull()]
    df_args = df_args[["scipy_function", "np_call", "np_args"]]

    total_np_calls = df_args.groupby(["np_call", "np_args"]).size()
    df_total_np_calls = pd.DataFrame({"total_np_calls": total_np_calls})

    num_scipy_funs_used = df_args.drop_duplicates().groupby(["np_call", "np_args"]).size()
    df_num_scipy_funs_used = pd.DataFrame({"num_scipy_funs_used_by": num_scipy_funs_used})

    df_call_stats = df_num_scipy_funs_used.join(df_total_np_calls, lsuffix="_caller", rsuffix="_other")
    df_call_stats = df_call_stats.sort_values(
        by=["num_scipy_funs_used_by", "total_np_calls"], ascending=False
    )

    my_out3 = out_dir + "scipy_np_arg_stats.csv"
    df_call_stats.to_csv(my_out3)
