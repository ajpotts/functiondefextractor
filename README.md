# Function Extractor

![Python application](https://github.com/philips-software/functiondefextractor/workflows/Python%20application/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/philips-software/functiondefextractor/branch/master/graph/badge.svg)](https://codecov.io/gh/philips-software/functiondefextractor)

Tool to extract the function definitions from the source code

It can be used to extract functions from,

- C  

- C++

- C#  

- Java  

- Python

- TypeScript

- JavaScript

Advantage of using such function extractions are,

- Resolving technical debt  

- Identify function similarity  

- Identify pattern check (Supresswarnings, Assert, etc...)
  
## Dependencies

- python 3.8 : 64 bit  

- python packages (xlrd, xlsxwriter, pandas)  

- third party packages [Ctags, grep]

## Installation
  
[INSTALL.md](INSTALL.md)

```sh
pip install functiondefextractor
```

## Usage & Configuration

### Code

- General usage with out options.

```sh
from functiondefextractor import core_extractor
out_put = core_extractor.extractor (r"path_to_repo/code")
print(out_put)
```

- To extract functions based on annotation.

```sh
from functiondefextractor import core_extractor
out_put = core_extractor.extractor (r"path_to_repo/code", annot="@Test")
print(out_put)
```

- To extract delta lines(+/-) from code based on annotation/key word.
Note: If user is unaware of complete annotation use this(annot with delta)
feature to extract functions else use the above feature.

```sh
from functiondefextractor import core_extractor
out_put = core_extractor.extractor
          (r"path_to_repo/code", annot="@SupressWarning", delta="5")
print(out_put)
```

- To analyse various patterns in the code based on given condition.
For example to search assert, suppress warnings patterns.

```sh
from functiondefextractor import condition_checker
out_put = core_extractor.check_condition
          ("@SupressWarning", r"path_to_excelfile/dataframe", "(")
print(out_put[0], out_put[1])
```

### Commandline

- General usage with out options to extract functions from repo.

```sh
>>>python -m functiondefextractor.extractor_cmd --p path/to/repo
```

- To analyse various patterns in the code based on given condition.

```sh
>>>python -m functiondefextractor.extractor_cmd
             --c "@Assert" --e path/to/excel/dataframe --s "("
```

- Help option can be found at,  

```sh
>>>python -m functiondefextractor.extractor_cmd -h
```

### Output
  
- Executing functiondefextractor to extract functions from
 command line would generate an output excel file which contains
 FileName_FunctionName in Unique ID column and extracted functions in Code column

- Using functiondefextractor to extract functions from code would return
 a dataframe with same content as excel file.

- When functiondefextractor is executed from script to analyse patterns in code,
 a tuple with 2 data frames would be generated which contains the requested pattern
 statements with their count in various functions and a pivot table of the
 same respectively.

## Contact

[MAINTAINERS.md](MAINTAINERS.md)  

## License

[License.md](LICENSE.md)
