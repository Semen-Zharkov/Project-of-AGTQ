from typing import Any
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import TextLoader
from langchain.chat_models.gigachat import GigaChat
from langchain.text_splitter import CharacterTextSplitter
from config_data.config import load_config, Config
from prompt_stats.summarize_stats import get_summarize_stats
from utils.create_prompts import create_prompt


@get_summarize_stats
def use_prompt_summarize(file_path: str, sys_prompt_path: str) -> Any:
    config: Config = load_config()

    giga: GigaChat = GigaChat(credentials=config.GIGA_CREDENTIALS, verify_ssl_certs=False)

    document = TextLoader(file_path, encoding='utf-8').load()
    split_docs = (CharacterTextSplitter(separator='\n', chunk_size=39000, chunk_overlap=0)
                  .split_documents(document))

    prompt = create_prompt(sys_prompt_path)
    chain = LLMChain(llm=giga, prompt=prompt)

    res = ''
    for i in split_docs:
        res += chain.run(i)

    return res
