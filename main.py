from langchain.chains import LLMChain
from langchain.document_loaders import TextLoader
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.chat_models.gigachat import GigaChat
from langchain.text_splitter import CharacterTextSplitter
from config import Config, load_config
from prompts.v1_prompt_template import prompt1


config: Config = load_config()
giga = GigaChat(credentials=config.GIGA_CREDENTIALS, verify_ssl_certs=False)


# С*** ПУТЬ К ФАЙЛУ
PATH = 'data/111.txt'

documents = TextLoader(PATH, encoding='utf-8').load()
split_docs = CharacterTextSplitter(chunk_size=3000, chunk_overlap=100).split_documents(documents)


# chain = LLMChain(llm=giga, prompt=prompt1, output_parser=SimpleJsonOutputParser())
# print(chain.run(text=split_docs, num=3))

res = ''
for doc in split_docs:
    temp = giga(
        prompt1.format_prompt(
            que_num=2,
            text=doc,
        ).to_messages()
    ).content
    res += f'{temp}\n\n'

print(res)
