import yaml
from src.utils.few_shot import FewShotPost
from src.utils.embedder import Embedder
from langchain_core.output_parsers import PydanticOutputParser
from src.Generator.schemas.generator_schema import GeneratedPosts

class PromptBuilder:
    def __init__(self, config_path: str) -> None:
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

    def build(self, event: dict, few_shot_posts: list[dict], format_instructions: str):
        system_identity = self.config["identity"]
        language_constraints = self.config["language_constraints"]
        format_constraints = self.config["linkedin_format_constraints"]
        # angles = self.config["angles"]

        examples = "\n\n".join(
            f"POST {i+1}:\n{post['text']}"
            for i, post in enumerate(few_shot_posts) 
        )

        arguments = "\n- ".join(event["arguments"])

        event_block = f"""
i) What happened: {event["what_happened"]}
ii) Global relevance: {event["why_global"]}
iii) Relevance for the Netherlands {event['why_nl']}
iv) Netherlands relevance {event["why_nl"]}
v) KickstartAI stance: {event["stance"]}
vi) Arguments supporting stance:
- {arguments} 
vii) Source: {event["source"]}
"""

        return f"""
1. Identity and role
{system_identity}

2. Task
Transform the structured input into 3 distinct LinkedIn post suggestions for KickstartAI.
Use the input as the factual basis for the posts. Do not invent facts or add unsupported claims.

2. Structured input
{event_block}

3. Language and writing constraints
{language_constraints}

4. LinkedIn formatting constraints
{format_constraints}

5. Historical KickstartAI LinkedIn examples
Use the following historical posts as style references only.
Focus on tone, flow, structure, and writing style.
Do not copy topic-specific wording or reuse sentences.
{examples}

6. Output Requirements
Return ONLY valid JSON.

{format_instructions}

"""


if __name__ == "__main__":
    # system components
    fs = FewShotPost()
    embedder = Embedder()
    builder = PromptBuilder("config/prompt_config_v3.yaml")

    #1. Fake Interpreter output (structured event)
    interpreter_output = {
        "what_happened": (
            "ING COO Risk Leon Dusée shared lessons on AI adoption in large organizations, "
            "explaining why many AI pilots stall and how organizations can scale AI successfully."
        ),

        "why_global": (
            "Organizations worldwide are struggling with the same challenge: moving AI from "
            "isolated experiments into real operational impact. The bottleneck is often not "
            "the technology itself, but focus, governance, ownership, and organizational alignment."
        ),

        "why_nl": (
            "Dutch organizations across sectors like banking, aviation, retail, education, "
            "and healthcare are trying to accelerate AI adoption while avoiding fragmented "
            "pilot initiatives. Collaboration between organizations is becoming increasingly important "
            "for scaling AI responsibly and effectively in the Netherlands."
        ),

        "why_kickstartai": (
            "KickstartAI helps organizations share practical lessons about AI implementation, "
            "scaling, governance, and adoption. The collaboration between companies like ING, "
            "KLM, NS, and Ahold Delhaize aims to accelerate AI adoption in the Netherlands "
            "while also creating broader societal impact."
        ),

        "stance": (
            "AI adoption should be treated as an organizational transformation challenge, "
            "not just a technology experiment. Focus, leadership, and investing in people "
            "are just as important as the AI systems themselves."
        ),

        "arguments": [
            "AI pilots often fail because organizations think about scaling and ownership too late",
            "Organizations need to focus on a limited number of high-impact use cases instead of experimenting everywhere",
            "AI requires employees and leaders to develop critical thinking and adaptability",
            "The biggest AI opportunities often require redesigning entire processes, not just optimizing individual tasks",
            "Cross-organizational collaboration helps accelerate AI adoption and societal impact in the Netherlands",
            "AI should not only improve productivity, but also contribute to areas like financial health, education, and healthcare"
        ],

        "source": "https://lnkd.in/eU8jqvT9"
    }

    #2. Convert event into structured text
    text = f"""
    What happened: {interpreter_output['what_happened']}
    Why global: {interpreter_output['why_global']}
    Why NL: {interpreter_output['why_nl']}
    Why KickstartAI: {interpreter_output['why_kickstartai']}
    Stance: {interpreter_output['stance']}
    Arguments: {', '.join(interpreter_output['arguments'])}
    """
    query_embedding = embedder.embed_text(text)

    # 4. Get similar posts
    few_shot_posts = fs.get_similar_posts(query_embedding, top_k=3)
    
    # 5. Buld prompt
    parser = PydanticOutputParser(
        pydantic_object=GeneratedPosts
    )
    format_instructions = parser.get_format_instructions()
    prompt = builder.build(interpreter_output, few_shot_posts, format_instructions)

    print(prompt)


# python3 -m src.Generator.prompt_builder_v3
