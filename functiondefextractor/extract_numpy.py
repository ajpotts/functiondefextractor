import pandas as pd
from core_extractor import extractor, get_report


def add_numpy_calls(df):
    df["np_calls"] = df["Code"].str.findall(r"np\.([\w\._]*)")
    df["num_np_calls"] = df["np_calls"].apply(lambda x: len(x))
    df["np_calls_w_args"] = df["Code"].str.findall(r"np\.([\w\_\.]+\(.*\))")

    return df


def add_library_functions(df: pd.DataFrame, lib_name: str, lib_path: str):
    df[lib_name + "_function"] = df["Uniq ID"].str.findall(
        lib_path
        + "([\w_/\.]*)"
    )
    df = df.explode(lib_name + "_function")
    return df


def get_and_write_np_stats(lib_name: str, lib_path: str, out_dir: str):
    df = extractor(lib_path)
    df = add_numpy_calls(df)
    df = add_library_functions(df, lib_name, lib_path)
    my_out = out_dir + lib_name + "_function_definitions.csv"
    df.to_csv(my_out)
    return df


def get_numpy_calls_stats(df, lib_name: str, out_dir: str):
    df_np_calls = df.explode("np_calls")[[lib_name + "_function", "np_calls"]]
    df_np_calls = df_np_calls[df_np_calls["np_calls"].notnull()]
    df_np_calls = df_np_calls.explode("np_calls")

    total_np_calls = df_np_calls.groupby("np_calls").size()
    df_total_np_calls = pd.DataFrame({"total_np_calls": total_np_calls})

    num_scipy_funs_used = df_np_calls.drop_duplicates().groupby("np_calls").size()
    df_num_scipy_funs_used = pd.DataFrame({"num_" + lib_name + "_funs_used_by": num_scipy_funs_used})

    df_call_stats = df_num_scipy_funs_used.join(df_total_np_calls, lsuffix="_caller", rsuffix="_other")
    df_call_stats = df_call_stats.sort_values(
        by=["num_" + lib_name + "_funs_used_by", "total_np_calls"], ascending=False
    )

    my_out = out_dir + lib_name + "_np_call_stats.csv"
    df_call_stats.to_csv(my_out)
    return df_call_stats


def get_numpy_arg_stats(df: pd.DataFrame, lib_name: str, out_dir: str):
    df_args = df.explode("np_calls_w_args")
    df_args = df_args[df_args["np_calls_w_args"].notnull()][[lib_name + "_function", "np_calls_w_args"]]
    df_args["np_call"] = df_args["np_calls_w_args"].str.findall(r"(^[\w\.]*)")
    df_args = df_args.explode("np_call")
    df_args["np_args"] = df_args["np_calls_w_args"].str.findall(r"([\w\._]*)\s*=[^=<>!]")
    df_args = df_args.explode("np_args")
    df_args = df_args[df_args["np_args"].notnull()]
    df_args = df_args[[lib_name + "_function", "np_call", "np_args"]]

    total_np_calls = df_args.groupby(["np_call", "np_args"]).size()
    df_total_np_calls = pd.DataFrame({"total_np_calls": total_np_calls})

    num_scipy_funs_used = df_args.drop_duplicates().groupby(["np_call", "np_args"]).size()
    df_num_scipy_funs_used = pd.DataFrame({"num_" + lib_name + "_funs_used_by": num_scipy_funs_used})

    df_call_stats = df_num_scipy_funs_used.join(df_total_np_calls, lsuffix="_caller", rsuffix="_other")
    df_call_stats = df_call_stats.sort_values(
        by=["num_" + lib_name + "_funs_used_by", "total_np_calls"], ascending=False
    )

    my_out = out_dir + lib_name + "_np_arg_stats.csv"
    df_call_stats.to_csv(my_out)

    return df_args


def get_fuction_stats(df: pd.DataFrame, lib_name: str, out_dir: str):
    df_abrev = df[[lib_name + "_function", "num_np_calls", "np_calls", "np_calls_w_args"]]
    df_abrev = df_abrev.sort_values(by=["num_np_calls"], ascending=False)
    df_abrev = df_abrev.reset_index()
    my_out = out_dir + lib_name + "_function_stats.csv"
    df_abrev.to_csv(my_out)
    return df_abrev


def run_stats(lib_name: str, lib_path:str, out_dir: str):
    df = get_and_write_np_stats(lib_name, lib_path, out_dir)
    df_call_stats = get_numpy_calls_stats(df, lib_name, out_dir)
    df_args = get_numpy_arg_stats(df, lib_name, out_dir)
    get_fuction_stats(df, lib_name, out_dir)
    return df


if __name__ == "__main__":
    out_dir = "/home/amandapotts/git/functiondefextractor/data/out/"
    scipy_path = "/home/amandapotts/git/scipy/scipy"
    pandas_path = "/home/amandapotts/git/pandas/pandas"
    arkouda_path = "/home/amandapotts/git/arkouda/arkouda"

    run_stats("arkouda", arkouda_path, out_dir)
    run_stats("scipy", scipy_path, out_dir)
    run_stats("pandas", pandas_path, out_dir)
