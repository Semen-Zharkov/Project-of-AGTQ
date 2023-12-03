from typing import Any
from random import randint
from langchain.chains import LLMChain
from langchain.document_loaders import TextLoader
from langchain.chat_models.gigachat import GigaChat
from langchain.text_splitter import CharacterTextSplitter
import time
from answering_questions.answer_questions import get_answer
from config_data.config import load_config, Config
from summarize.summarize_map_reduce import use_map_reduce
from summarize.summarize_with_prompt import use_prompt_summarize
from utils.create_prompts import create_prompt
from prompt_stats.get_stats import get_statistics
from dita_case.scrap_files import get_dita_docs
from answering_questions.docs_filtration import filter_docs


@get_statistics()
def main(file_path: str, system_prompt_path: str, user_prompt_path: str, que_num: int, dita: int) -> Any:
    config: Config = load_config()

    giga: GigaChat = GigaChat(credentials=config.GIGA_CREDENTIALS, verify_ssl_certs=False)

    if dita == 1:
        split_docs = get_dita_docs(min_doc_length=10000)
    else:
        document = TextLoader(file_path, encoding='utf-8').load()
        split_docs = (CharacterTextSplitter(separator='\n', chunk_size=39000, chunk_overlap=0)
                      .split_documents(document))

    prompt = create_prompt(system_prompt_path, user_prompt_path)

    chain = LLMChain(llm=giga, prompt=prompt)

    picked_items = []
    if sum(len(i.page_content) for i in split_docs) > 39000:
        res = ''
        for i in range(que_num):
            rnd = randint(0, len(split_docs) - 1)
            while rnd in picked_items:
                rnd = randint(0, len(split_docs) - 1)
            res += chain.run(num=1, text=split_docs[rnd]) + '\n\n'
        return res
    else:
        return chain.run(num=que_num, text=split_docs)


if __name__ == '__main__':
    SYS_PROMPT_PATH = 'prompts/generate_question_prompt.yaml'
    USR_PROMPT_PATH = ''
    PATH = 'data/it.txt'
    questions_number = 3
    # print(main(PATH, SYS_PROMPT_PATH, USR_PROMPT_PATH, questions_number, dita=1))

    """Блок ответов на вопросы"""
    # questions_for_ittxt = ['Кто такой Профессор Ка?', 'Что изобрёл этот ваш Профессор Ка?']
    # questions_for_crimetxt = ['Кто такой Раскольников?']
    # questions_for_dita = ['Какие возможности предоставляет DATAPK ITM?']
    # for q_num, que in enumerate(questions_for_dita, start=1):
    #     result = 'Ответ не найден :('
    #     start_time = time.time()
    #     for i in range(1, 487):
    #         result = get_answer('data/it.txt', 'prompts/qna_system.yaml',
    #                             'prompts/qna_user.yaml', que, i, dita=1)
    #         if result != 'Я не могу ответить на вопрос на основе информации. Попробуйте переформулировать вопрос.':
    #             print(f'Ответ найден за {time.time() - start_time} секунд в документе №{i}')
    #             break
    #     print(f'Вопрос {q_num}: {que}\nОтвет: {result}\n')


