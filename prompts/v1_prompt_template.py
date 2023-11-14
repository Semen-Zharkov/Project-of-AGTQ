from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, load_prompt

template = "Ты ассистент, который создает тесты на знание материала"
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = """
                 {text}
                 создай тест на знание текста из {num} вопросов с 4 вариантами ответов в каждом. Вопросы не нумеруй
                 """

human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)


prompt1 = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(
            prompt=load_prompt('prompts/generate_question_prompt.yaml')
        )
    ]
)
