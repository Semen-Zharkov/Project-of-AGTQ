from functools import wraps
import csv
import yaml


def is_format_followed(string: str, que_num: int) -> tuple[bool, bool]:
    string_list = string.split('\n\n')
    string_list = list(filter(None, map(lambda x: list(filter(None, x.split('\n'))), string_list)))
    is_corr_num = True if len(string_list) == que_num else False

    for enum, que in enumerate(string_list, start=1):
        for i in que:
            if i[0] != str(enum) and i[0] not in 'abcd':
                return False, is_corr_num
    return True, is_corr_num


def read_yaml(path_yaml: str) -> str:
    with open(path_yaml, encoding='utf-8') as fh:
        dict_data = yaml.safe_load(fh)
        template = dict_data['template']
        return template


def get_doc_length(path_doc: str) -> int:
    with open(path_doc, 'r', encoding='utf-8') as file:
        return len(file.read())


def get_statistics():
    def func_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            templ = read_yaml(args[1])
            res = func(*args, **kwargs)
            with open('prompt_stats/statistics.csv', 'a+', encoding='cp1251', newline='') as stats:
                writer = csv.writer(stats, delimiter=';')
                writer.writerow(
                    [args[0], args[1], get_doc_length(args[0]), *is_format_followed(res, args[2]), templ, res]
                )
            return res

        return wrapper
    return func_decorator
