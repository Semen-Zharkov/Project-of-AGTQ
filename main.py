from langchain.chains import LLMChain
from langchain.document_loaders import TextLoader
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.chat_models.gigachat import GigaChat
from langchain.text_splitter import CharacterTextSplitter
from config import Config, load_config


config: Config = load_config()
giga = GigaChat(credentials=config.GIGA_CREDENTIALS, verify_ssl_certs=False)


template = "Ты ассистент, который создает тесты на знание материала"
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = """
                 {text}
                 создай тест на знание текста из {num} вопросов с 4 вариантами ответов в каждом
                 """

human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)


# С*** ПУТЬ К ФАЙЛУ
PATH = 'data/111.txt'

documents = TextLoader(PATH, encoding='utf-8').load()
# split_docs = CharacterTextSplitter(chunk_size=3000, chunk_overlap=100).split_documents(documents)
#
#
# chain = LLMChain(llm=giga, prompt=chat_prompt, output_parser=SimpleJsonOutputParser())
# print(chain.run(text=split_docs, num=3))


res = giga(
    chat_prompt.format_prompt(
        text=documents,
        num=3,
    ).to_messages()
)

print(res.content)
