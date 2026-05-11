from stores.llm.tempelate.locales.en import build_prompt as en_prompt
from stores.llm.tempelate.locales.ar import build_prompt as ar_prompt


class PromptTemplate:
    _builders = {
        "en": en_prompt,
        "ar": ar_prompt,
    }

    @classmethod
    def build(cls, context: str, query: str, language: str = "en") -> str:
        builder = cls._builders.get(language, en_prompt)
        return builder(context=context, query=query)
