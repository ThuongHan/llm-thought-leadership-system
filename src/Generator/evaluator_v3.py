from src.utils.llm_clients import llm
from src.Generator.schemas.evaluation_schema import EvaluationResult
import yaml
import json
from langchain_core.output_parsers import PydanticOutputParser
from pathlib import Path

class Evaluator():
    def __init__(self, config_path: str):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.llm = llm
        self.parser = PydanticOutputParser(
            pydantic_object=EvaluationResult
        )

    def evaluate(self, LinkedIn_posts: str, few_shot_posts: list[dict]):

        identity = self.config["identity"]
        rubric = self.config["rubrics"]
        output_rules = self.config["evaluation_output_requirements"]
        reference_block = "\n\n".join(
            f"REFERENCE {i+1}:\n{post['text']}"
            for i, post in enumerate(few_shot_posts)
        )
        format_instructions = self.parser.get_format_instructions()
        output_format = f"""
FORMAT:
{output_rules["format"]}

PER DIMENSION OUTPUT
{output_rules["per_dimension_output"]}

COMMENT RULES:
{output_rules["comment_style_rules"]}
"""

        prompt = f"""
{identity}

RUBRIC:
{rubric}

HISTORICAL REFERENCE POSTS:
{reference_block}

POSTS TO EVALUATE:
{LinkedIn_posts}

{format_instructions}
"""
        
        response = self.llm.invoke(prompt)
        parsed = self.parser.parse(response.content)
        saved_path = self.save_to_file(parsed)
        print(f"\nSaved to: {saved_path}")
        return parsed
    
    def save_to_file(self, result: dict, filename: str = "evaluated_posts.json"):

        output_dir = Path("generated_evaluations")
        output_dir.mkdir(exist_ok=True)

        data = { 
            "evaluations": [
                evaluation.model_dump()
                for evaluation in result.evaluations
            ]
        }

        file_path = output_dir / filename

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
        
        return str(file_path)


if __name__ == "__main__":

    with open("generated_posts/generated_posts.json", "r") as file:
        data = json.load(file)

    posts = data["posts"]
    few_shot = data["few_shot_examples"]

    evaluator = Evaluator("config/evaluation_config_v3.yaml")

    result = evaluator.evaluate(
        LinkedIn_posts=json.dumps(posts, indent=2),
        few_shot_posts=few_shot
    )

    print("\nEVALUATION RESULT:\n")
    print(result)

# python3 -m src.Generator.evaluator_v3