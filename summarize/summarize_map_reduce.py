from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import TextLoader
from langchain.chat_models.gigachat import GigaChat
from langchain.text_splitter import CharacterTextSplitter
from config import load_config, Config
from prompt_stats.summarize_stats import get_summarize_stats


@get_summarize_stats
def get_summary(file_path):
    config: Config = load_config()

    giga: GigaChat = GigaChat(credentials=config.GIGA_CREDENTIALS, verify_ssl_certs=False)

    document = TextLoader(file_path, encoding='utf-8').load()
    split_docs = (CharacterTextSplitter(separator='\n', chunk_size=5000, chunk_overlap=500)
                  .split_documents(document))

    chain = load_summarize_chain(giga, chain_type="map_reduce")
    res = chain.run(split_docs)

    print("\n\n===")
    return res
