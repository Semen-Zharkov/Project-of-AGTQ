from functools import wraps
import time
from prompt_stats.get_stats import get_doc_length


def get_summ_tokens(doc_length: int, returned: str) -> int:
    sm = doc_length + len(returned)
    return (sm // 3 + sm // 2) // 2


def get_summarize_stats(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        res_time = time.time() - start_time
        chars = get_doc_length(args[0])
        print(f'[INFO] Выполнение заняло {res_time} секунд | Было потрачено {get_summ_tokens(chars, res)} токенов (Текст {chars} символов)')
        return res

    return wrapper
