from langchain.chains import LLMChain
from langchain.document_loaders import TextLoader
from langchain.chat_models.gigachat import GigaChat
from langchain.text_splitter import CharacterTextSplitter
from config import load_config, Config
from utils.create_prompts import create_prompt
from prompt_stats.get_stats import get_statistics
from random import randint


# @get_statistics()
def main(file_path: str, prompt_path: str, que_num: int):
    config: Config = load_config()

    giga: GigaChat = GigaChat(credentials=config.GIGA_CREDENTIALS, verify_ssl_certs=False)

    document = TextLoader(file_path, encoding='utf-8').load()
    split_docs = (CharacterTextSplitter(separator='\n', chunk_size=39000, chunk_overlap=0)
                  .split_documents(document))

    prompt = create_prompt(prompt_path)
    chain = LLMChain(llm=giga, prompt=prompt)

    if sum(len(i.page_content) for i in split_docs) > 39000:
        res = ''
        for i in range(que_num):
            res += chain.run(num=1, text=split_docs[randint(0, len(split_docs)) - 1]) + '\n\n'
        return res
    else:
        chain.run(num=que_num, text=split_docs)


if __name__ == '__main__':
    PROMPT_PATH = 'prompts/generate_question_prompt.yaml'
    PATH = 'data/crime3.txt'
    questions_number = 3
    print(main(PATH, PROMPT_PATH, questions_number))
