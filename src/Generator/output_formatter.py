from langchain_core.messages import AIMessage
import re

class OutputFormatter():

    def format(self, reponse: AIMessage, few_shot_examples: list[dict]) -> dict:
        content = reponse.content.strip()
        hashtags = re.findall(r"#\w+", content)

        return {
            "posts": content,
            "hashtags": hashtags,
            "few_shot_examples": few_shot_examples,
            "post_count": content.count("**Post")
        }
    


    