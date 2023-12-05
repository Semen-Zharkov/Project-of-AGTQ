from typing import Any
from random import randint
from langchain.chains import LLMChain
from langchain.chat_models.gigachat import GigaChat
from config_data.config import load_config, Config
from utils.create_prompts import create_prompt
from utils.split_docs import get_docs_list
from utils.output_parser import parse_output
from prompt_stats.get_stats import get_statistics
from dita_case.scrap_files import get_dita_docs
from data.test_questions.test_questions import *
from answering_questions.answer_questions import get_answer


@get_statistics()
def main(file_path: str, system_prompt_path: str, user_prompt_path: str, que_num: int, dita: int) -> Any:
    config: Config = load_config()

    giga: GigaChat = GigaChat(credentials=config.GIGA_CREDENTIALS, verify_ssl_certs=False)

    if dita == 1:
        split_docs = get_dita_docs(min_doc_length=10000)
    else:
        split_docs = get_docs_list(file_path)

    prompt = create_prompt(system_prompt_path, user_prompt_path)

    chain = LLMChain(llm=giga, prompt=prompt)

    if sum(len(i.page_content) for i in split_docs) > 39000:
        res = ''
        for i in range(que_num):
            rnd = randint(0, len(split_docs) - 1)
            res += chain.run(num=1, text=split_docs[rnd]) + '\n\n'
        return res
    else:
        result = chain.run(num=que_num, text=split_docs)
        return parse_output(result)


if __name__ == '__main__':
    SYS_PROMPT_PATH = 'prompts/generate_question_prompt.yaml'
    USR_PROMPT_PATH = ''
    PATH = 'data/crime6.txt'
    questions_number = 5
    print(main(PATH, SYS_PROMPT_PATH, USR_PROMPT_PATH, questions_number, dita=0))

    """Блок ответов на вопросы"""
    # for answer in get_answer('data/crime6.txt', 'prompts/qna_system.yaml',
    #                          'prompts/qna_user.yaml', questions_for_crimetxt, dita=0):
    #     print(answer)

