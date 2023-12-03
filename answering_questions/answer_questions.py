from typing import Any
from langchain.chains import LLMChain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from utils.create_prompts import create_prompt
from config_data.config import load_config, Config
from langchain.chat_models.gigachat import GigaChat
from dita_case.scrap_files import get_dita_docs


def get_answer(file_path: str, prompt_system: str, prompt_user: str, que: str, doc_num: int, dita: int) -> Any:
    config: Config = load_config()

    giga: GigaChat = GigaChat(credentials=config.GIGA_CREDENTIALS, verify_ssl_certs=False)

    if dita == 1:
        split_docs = get_dita_docs()[doc_num]
    else:
        document = TextLoader(file_path, encoding='utf-8').load()
        split_docs = (CharacterTextSplitter(separator='\n', chunk_size=39000, chunk_overlap=0)
                      .split_documents(document))

    prompt = create_prompt(prompt_system, prompt_user)

    chain = LLMChain(llm=giga, prompt=prompt)
    return chain.run(question=que, summaries=split_docs)

