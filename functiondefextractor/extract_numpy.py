import os
import re

import pandas as pd
from core_extractor import extractor, get_report


def add_numpy_calls(df):
    df["np_calls"] = df["Code"].str.findall(r"np\.([\w\._]*)")
    df["num_np_calls"] = df["np_calls"].apply(lambda x: len(x))
    df["np_calls_w_args"] = df["Code"].str.findall(r"np\.([\w\_\.]+\(.*\))")

    return df


def add_library_functions(df: pd.DataFrame, lib_name: str, lib_path: str):
    df[lib_name + "_function"] = df["Uniq ID"].str.findall(lib_path + "([\w_/\.]*)")
    df = df.explode(lib_name + "_function")
    df[lib_name + "_function"] = df[lib_name + "_function"].str.replace("(\.$)", "")
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
    df_np_calls[lib_name + "_function"] = df_np_calls[lib_name + "_function"].str.rstrip(r"\.")
    df_np_calls = df_np_calls.explode("np_calls")

    df_np_calls = df_np_calls.rename(columns={"np_calls": "function_np"})
    df_np_calls["function_np"] = df_np_calls["function_np"].str.rstrip(r"\.")

    df_call_stats = (
        df_np_calls.groupby(["function_np"])
        .agg(["count", "nunique"])
        .rename(columns={"count": "total_np_calls", "nunique": "num_" + lib_name + "_funs_used_by"})
    )

    df_call_stats.columns = df_call_stats.columns.get_level_values(1)
    df_call_stats["used_by_" + lib_name] = df_call_stats["total_np_calls"] > 0

    df_call_stats = df_call_stats.sort_values(
        by=[
            "num_" + lib_name + "_funs_used_by",
            "total_np_calls",
        ],
        ascending=False,
    )

    df_call_stats = df_call_stats.reset_index(col_fill=["function_np", "np_arg"])

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

    df_call_stats = (
        df_args.groupby(["np_call", "np_args"])
        .agg(["count", "nunique"])
        .rename(columns={"count": "total_np_calls", "nunique": "num_" + lib_name + "_funs_used_by"})
    )
    df_call_stats.columns = df_call_stats.columns.get_level_values(1)

    df_call_stats = df_call_stats.sort_values(
        by=[
            "num_" + lib_name + "_funs_used_by",
            "total_np_calls",
        ],
        ascending=False,
    )
    df_call_stats = df_call_stats.reset_index(col_fill=["function_np", "np_arg"])

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


def run_stats(lib_name: str, lib_path: str, out_dir: str):
    df = get_and_write_np_stats(lib_name, lib_path, out_dir)
    df_call_stats = get_numpy_calls_stats(df, lib_name, out_dir)
    df_args = get_numpy_arg_stats(df, lib_name, out_dir)
    get_fuction_stats(df, lib_name, out_dir)
    return df, df_call_stats, df_args


def enhance_numpy_api():
    df = pd.read_csv(np_api_sheet, names=["np"], header=None)
    df["np"] = df["np"].str.strip()

    df["function"] = df["np"].str.extract(r"numpy\.([\w\._]*)", expand=True).astype(str)
    df["function_name"] = df["np"].str.extract(r"([\w_]*$)", expand=True).astype(str)
    df["numpy_api_link"] = (
        "https://numpy.org/doc/stable/reference/generated/" + df["np"].astype(str) + ".html"
    )
    df = df.drop_duplicates()
    my_out = out_dir + "numpy_api_enhanced.csv"
    df.to_csv(my_out)
    return df


def extract_api(filename):
    p = re.compile(r"title=\"(arkouda\.[\w\._]*)")
    with open(filename, mode="rt", encoding="utf-8") as docFile:
        doc = docFile.read()
        names = re.findall(p, doc)
    return names


def get_arkouda_api_from_docs(rootdir: str):
    regex = re.compile("(.*index.html)")

    api_list = []
    for root, dirs, files in os.walk(rootdir):
        if len(dirs) > 0:
            for dir in dirs:
                api = extract_api(root + "/" + dir + "/index.html")
                api_list.extend(api)
        else:
            api = extract_api(root + "/index.html")
            api_list.extend(api)
    return api_list


def get_arkouda_api_df_from_docs(rootdir: str, out_dir: str):
    api_list = get_arkouda_api_from_docs(rootdir)
    df = pd.DataFrame()
    df["ak"] = api_list
    df["ak"] = df["ak"].str.strip()
    df = df.drop_duplicates()
    df["function"] = df["ak"].str.extract(r"arkouda\.([\w\._]*)", expand=True).astype(str)
    df["function_name"] = df["ak"].str.extract(r"([\w_]*$)", expand=True).astype(str)
    df[
        "arkouda_api_link"
    ] = "file:///home/amandapotts/git/arkouda/docs/autoapi/arkouda/alignment/index.html#" + df[
        "ak"
    ].astype(
        str
    )
    df = df.reset_index()
    my_out = out_dir + "arkouda_api_enhanced.csv"
    df.to_csv(my_out)
    return df


def generate_api_comparision(np_df, ak_df):
    api_comparison = np_df.merge(
        ak_df, on=["function_name"], how="outer", suffixes=("_np", "_ak")
    ).reset_index()
    api_comparison = api_comparison[
        ["function_name", "np", "function_np", "numpy_api_link", "ak", "function_ak", "arkouda_api_link"]
    ]
    api_comparison["partial_ak_coverage"] = api_comparison["function_ak"].notnull()
    api_comparison = api_comparison.drop_duplicates()

    my_out = out_dir + "np_ak_api_comparison.csv"
    api_comparison.to_csv(my_out)
    return api_comparison


def merge_arkoua_api_onto(libname: str, df_call_stats: pd.DataFrame, api_comparison: pd.DataFrame):
    df_call_stats["function_np"] = df_call_stats["function_np"].astype(str)
    api_comparison["function_np"] = api_comparison["function_np"].astype(str)
    merged_df = df_call_stats.merge(api_comparison, on=["function_np"], how="outer")
    merged_df = merged_df[(merged_df["function_np"] != "nan") | (merged_df["function_name"] == "nan")]
    merged_df = merged_df.reset_index()

    my_out = out_dir + libname + "_api_comparison.csv"
    merged_df.to_csv(my_out)
    return merged_df


def get_coverage_stats(lib_name: str, df: pd.DataFrame):
    stats = df.groupby(["partial_ak_coverage"]).agg(["count", "nunique", "sum"])
    stats = stats[
        [
            ("function_np", "nunique"),
            ("total_np_calls", "sum"),
            ("num_" + lib_name + "_funs_used_by", "sum"),
            ("used_by_" + lib_name, "sum"),
        ]
    ]
    stats.columns = stats.columns.get_level_values(0) + "_" + stats.columns.get_level_values(1)
    stats = stats.rename(
        columns={
            "function_np_nunique": "total_unique_numpy_functions",
            "used_by_"+lib_name+"_sum": "total_unique_numpy_functions_used_by_" + lib_name,
            "total_np_calls_sum": "total_np_calls",
            "num_"+lib_name+"_funs_used_by_sum": "num_"+lib_name+"_funs_used",
        }
    )

    stats = stats[
        [
            "total_unique_numpy_functions",
            "total_np_calls",
            "num_" + lib_name + "_funs_used",
            "total_unique_numpy_functions_used_by_" + lib_name,
        ]
    ]

    my_out = out_dir + lib_name + "_coverage_stats.csv"
    stats.to_csv(my_out)
    return stats


if __name__ == "__main__":
    out_dir = "/home/amandapotts/git/functiondefextractor/data/out/"
    scipy_path = "/home/amandapotts/git/scipy/scipy"
    pandas_path = "/home/amandapotts/git/pandas/pandas"
    arkouda_path = "/home/amandapotts/git/arkouda/arkouda"
    nltk_path = "/home/amandapotts/git/nltk/nltk"

    np_api_sheet = "/home/amandapotts/git/functiondefextractor/data/numpy_api/np.csv"
    arkouda_docs_path = "/home/amandapotts/git/arkouda/docs/autoapi/arkouda/"

    np_df = enhance_numpy_api()
    ak_df = get_arkouda_api_df_from_docs(arkouda_docs_path, out_dir)
    api_comparison = generate_api_comparision(np_df, ak_df)

    arkouda_df, arkouda_df_call_stats, arkouda_df_args = run_stats("arkouda", arkouda_path, out_dir)
    scipy_df, scipy_df_call_stats, scipy_df_args = run_stats("scipy", scipy_path, out_dir)
    pandas_df, pandas_df_call_stats, pandas_df_args = run_stats("pandas", pandas_path, out_dir)
    nltk_df, nltk_df_call_stats, nltk_df_args = run_stats("nltk", nltk_path, out_dir)

    scipy_api_comparison = merge_arkoua_api_onto("scipy", scipy_df_call_stats, api_comparison)
    pandas_api_comparison = merge_arkoua_api_onto("pandas", pandas_df_call_stats, api_comparison)
    nltk_api_comparison = merge_arkoua_api_onto("nltk", nltk_df_call_stats, api_comparison)

    get_coverage_stats("scipy", scipy_api_comparison)
    get_coverage_stats("pandas", pandas_api_comparison)
    get_coverage_stats("nltk", nltk_api_comparison)
