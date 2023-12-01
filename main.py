from typing import Any
from langchain.chains import LLMChain
from langchain.document_loaders import TextLoader
from langchain.chat_models.gigachat import GigaChat
from langchain.text_splitter import CharacterTextSplitter
from config import load_config, Config
from summarize.summarize_map_reduce import get_summary
from utils.create_prompts import create_prompt
from prompt_stats.get_stats import get_statistics
from random import randint


@get_statistics()
def main(file_path: str, system_prompt_path: str, user_prompt_path: str, que_num: int) -> Any:
    config: Config = load_config()

    giga: GigaChat = GigaChat(credentials=config.GIGA_CREDENTIALS, verify_ssl_certs=False)

    document = TextLoader(file_path, encoding='utf-8').load()
    split_docs = (CharacterTextSplitter(separator='\n', chunk_size=39000, chunk_overlap=0)
                  .split_documents(document))

    prompt = create_prompt(system_prompt_path, user_prompt_path)

    chain = LLMChain(llm=giga, prompt=prompt)

    if sum(len(i.page_content) for i in split_docs) > 39000:
        res = ''
        for i in range(que_num):
            res += chain.run(num=1, text=split_docs[randint(0, len(split_docs)) - 1]) + '\n\n'
        return res
    else:
        return chain.run(num=que_num, text=split_docs)


if __name__ == '__main__':
    SYS_PROMPT_PATH = 'prompts/generate_question_prompt.yaml'
    USR_PROMPT_PATH = ''
    PATH = 'data/it.txt'
    questions_number = 1
    print(main(PATH, SYS_PROMPT_PATH, USR_PROMPT_PATH, questions_number))


