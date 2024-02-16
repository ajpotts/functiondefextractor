import pandas as pd
from core_extractor import extractor, get_report
import re
import os
import re


def add_numpy_calls(df):
    df["np_calls"] = df["Code"].str.findall(r"np\.([\w\._]*)")
    df["num_np_calls"] = df["np_calls"].apply(lambda x: len(x))
    df["np_calls_w_args"] = df["Code"].str.findall(r"np\.([\w\_\.]+\(.*\))")

    return df


def add_library_functions(df: pd.DataFrame, lib_name: str, lib_path: str):
    df[lib_name + "_function"] = df["Uniq ID"].str.findall(lib_path + "([\w_/\.]*)")
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

    df_call_stats = df_num_scipy_funs_used.join(
        df_total_np_calls, lsuffix="_num_functs", rsuffix="_total"
    )
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


def run_stats(lib_name: str, lib_path: str, out_dir: str):
    df = get_and_write_np_stats(lib_name, lib_path, out_dir)
    df_call_stats = get_numpy_calls_stats(df, lib_name, out_dir)
    df_args = get_numpy_arg_stats(df, lib_name, out_dir)
    get_fuction_stats(df, lib_name, out_dir)
    return df, df_call_stats, df_args


def enhance_api():
    df = pd.read_csv(np_api_sheet, names=["np"], header=None)
    print(df)
    df["np"] = df["np"].str.strip()
    df["np_function"] = df["np"].str.extract(r"numpy\.([\w\._]*)", expand=True)
    df["api_link"] = "https://numpy.org/doc/stable/reference/generated/" + df["np"].astype(str) + ".html"
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

    api_list = list()
    for root, dirs, files in os.walk(rootdir):
        if len(dirs) > 0:
            for dir in dirs:
                api = extract_api(root + "/" + dir + "/index.html")
                api_list.append(api)
        else:
            api = extract_api(root + "/index.html")
            api_list.append(api)
    return api_list


def get_arkouda_api_df_from_docs(rootdir: str, out_dir:str):
    api_list = get_arkouda_api_from_docs(rootdir)
    df = pd.DataFrame()
    df["ak"] = api_list
    my_out = out_dir + "arkouda_api_enhanced.csv"
    df.to_csv(my_out)
    return df



if __name__ == "__main__":
    out_dir = "/home/amandapotts/git/functiondefextractor/data/out/"
    scipy_path = "/home/amandapotts/git/scipy/scipy"
    pandas_path = "/home/amandapotts/git/pandas/pandas"
    arkouda_path = "/home/amandapotts/git/arkouda/arkouda"

    np_api_sheet = "/home/amandapotts/git/functiondefextractor/data/numpy_api/np.csv"

    # run_stats("arkouda", arkouda_path, out_dir)
    # df, df_call_stats, df_args = run_stats("scipy", scipy_path, out_dir)
    # run_stats("pandas", pandas_path, out_dir)

    df_numpy_api = enhance_api()

    rootdir = "/home/amandapotts/git/arkouda/docs/autoapi/arkouda/"
    api_list = get_arkouda_api_from_docs(rootdir)
    print(api_list)

    get_arkouda_api_df_from_docs(rootdir, out_dir)

    # df_test = df_call_stats.join(df_numpy_api, lsuffix="_num_functs", rsuffix="_total")
    #
    # my_out = out_dir + "test.csv"
    # df_test.to_csv(my_out)

# df = pd.DataFrame(columns=["Object", "Points", "Length"])
# with open('textfile.txt', 'r') as input, open('new_textfile.txt', 'w') as output:
#     for line in input:
#         if "OBJECT" in line.strip():
#             object_num = int(line.strip()[7:])
#
#         if "CONTOUR" in line.strip():
#             index_points = line.split(" ").index("points") - 1
#             index_length = line.split(" ").index("length") + 2
#             df.loc[len(df)] = {"Object": object_num, "Points": int(line.split(" ")[index_points]),
#                                "Length": "{:.2E}".format(float(line.split(" ")[index_length]))}
#
# input.close()
# output.close()
#
# print(df)
