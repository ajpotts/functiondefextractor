import os
import re

import numpy as np
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
    df[lib_name + "_function"] = df[lib_name + "_function"].astype(str)
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
    df_np_calls["np_calls"] = df_np_calls["np_calls"].astype(str)
    df_np_calls = df_np_calls[df_np_calls["np_calls"].notnull()]
    df_np_calls[lib_name + "_function"] = df_np_calls[lib_name + "_function"].str.rstrip(r"\.")
    # df_np_calls = df_np_calls.explode("np_calls")

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
    df_args["np_call"] = df_args["np_call"].astype(str)
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
    return merged_df


def merge_and_write_arkoua_api_onto(
    libname: str, df_call_stats: pd.DataFrame, api_comparison: pd.DataFrame, out_dir: str
):
    merged_df = merge_arkoua_api_onto(libname, df_call_stats, api_comparison)
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
            "used_by_" + lib_name + "_sum": "total_unique_numpy_functions_used_by_" + lib_name,
            "total_np_calls_sum": "total_np_calls",
            "num_" + lib_name + "_funs_used_by_sum": "num_" + lib_name + "_funs_used",
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


def run_all(lib_name: str, code_path: str, oud_dir: str, api_comparison: pd.DataFrame):
    df, df_call_stats, df_args = run_stats(lib_name, code_path, out_dir)
    lib_api_comparison = merge_and_write_arkoua_api_onto(
        lib_name, df_call_stats, api_comparison, out_dir
    )
    coverage_stats = get_coverage_stats(lib_name, lib_api_comparison)
    lib_coverage_stats = get_lib_coverage(lib_name, df, api_comparison)
    return df, df_call_stats, df_args, lib_api_comparison, coverage_stats, lib_coverage_stats


def get_lib_coverage(lib_name: str, df: pd.DataFrame, api_comparison: pd.DataFrame):
    df = df[[lib_name + "_function", "np_calls"]]
    df = df.explode("np_calls")
    df["np_calls"] = df["np_calls"].astype(str)
    df = df.drop_duplicates()
    df["function_np"] = df["np_calls"]

    merged_df = merge_arkoua_api_onto(lib_name, df, api_comparison)
    merged_df = merged_df.sort_values(
        by=[
            lib_name + "_function",
        ],
        ascending=True,
    )

    merged_df = merged_df[[lib_name + "_function", "np_calls", "function_ak", "partial_ak_coverage"]]
    merged_df = merged_df.drop_duplicates()
    merged_df["np_calls"] = merged_df["np_calls"].astype(str)
    merged_df["function_ak"] = merged_df["function_ak"].astype(str)

    grouped_df = merged_df.groupby([lib_name + "_function"]).agg(["sum", np.array]).copy(deep=True)
    grouped_df.columns = (
        grouped_df.columns.get_level_values(0) + "_" + grouped_df.columns.get_level_values(1)
    )
    grouped_df = grouped_df[["partial_ak_coverage_sum", "np_calls_array", "function_ak_array"]]
    grouped_df["partial_ak_coverage_sum"] = 1 * grouped_df["partial_ak_coverage_sum"]
    grouped_df["function_ak_array"] = grouped_df["function_ak_array"].apply(lambda x: x[x != "nan"])
    grouped_df["np_calls_array"] = grouped_df["np_calls_array"].apply(lambda x: x[x != "nan"])
    grouped_df["np_calls_count"] = grouped_df["np_calls_array"].apply(lambda x: len(x))
    grouped_df["coverage"] = (grouped_df["partial_ak_coverage_sum"] + 0.0001) / (
        grouped_df["np_calls_count"] + 0.0001
    )
    grouped_df["coverage"] = grouped_df["coverage"].astype(float).round(decimals=2)

    grouped_df = grouped_df[
        [
            "coverage",
            "np_calls_count",
            "partial_ak_coverage_sum",
            "np_calls_array",
            "function_ak_array",
        ]
    ]

    my_out = out_dir + lib_name + "_library_coverage_stats.csv"
    grouped_df.to_csv(my_out)
    return grouped_df


if __name__ == "__main__":
    git_dir = "/home/amandapotts/git/"
    out_dir = "/home/amandapotts/git/functiondefextractor/data/out/"
    np_api_sheet = "/home/amandapotts/git/functiondefextractor/data/numpy_api/np.csv"
    arkouda_docs_path = "/home/amandapotts/git/arkouda/docs/autoapi/arkouda/"
    arkouda_path = "/home/amandapotts/git/arkouda/arkouda/"

    lib_names = [
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("nltk", "nltk"),
        ("scikit-learn", "sklearn"),
        ("statsmodels", "statsmodels"),
        ("networkx", "networkx"),
    ]

    np_df = enhance_numpy_api()
    ak_df = get_arkouda_api_df_from_docs(arkouda_docs_path, out_dir)
    api_comparison = generate_api_comparision(np_df, ak_df)

    arkouda_df, arkouda_df_call_stats, arkouda_df_args = run_stats("arkouda", arkouda_path, out_dir)

    for lib_dir, lib_name in lib_names:
        lib_path = git_dir + lib_dir + "/" + lib_name
        run_all(lib_name, lib_path, out_dir, api_comparison)
