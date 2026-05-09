import yaml
from src.utils.few_shot import FewShotPost
from src.utils.embedder import Embedder
# event here should be a structured output from the interpreter
# event =  { "What happened" :,
    #        "Why does it matter (globally and NL)" :,
    #        "Why does it matter for KickstartAI" :,
    #        "Key stance / opinion":,
    #        "Supporting arguments":
    #       }


identity_block = """IDENTITY:

You are writing AS KickstartAI.

KickstartAI is a Dutch AI ecosystem initiative that accelerates real-world AI adoption with partners like ING, KLM, NS, and Ahold Delhaize.

You are NOT an observer.
You are NOT summarizing KickstartAI.
You ARE KickstartAI.

Write in first person plural ("we", "our", "us") where appropriate."""

class PromptBuilder:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def format_config(self):
        voice = ", ".join(self.config["voice"]["identity"])
        tone  = ", ".join(self.config["tone"]["attributes"])
        style = self.config["writing_style"]

        return f"""
VOICE: {voice}

TONE: {tone}

STYLE:
- Language: {style['language']}
- Brand: {style['brand_name']}
- Active voice: {style['voice']}
- Constraints:
  - No em dashes
  - No AI phrasing
  - Avoid redundancy

STRUCTURE:
- Max paragraph length: {style['structure']['max_paragraph_length_lines']} lines
- Prefer short sentences
        """
    def build(self, event: dict, few_shot_posts: list[dict]) -> str:
        
        config_block = self.format_config()
        examples = "\n\n".join(
            f"EXAMPLE {i+1}:\n{post['text']}"
            for i, post in enumerate(few_shot_posts)
        )

        event_block = f"""
What happened: {event['what_happened']}

Global relevance: {event['why_global']}
NL relevance: {event['why_nl']}
KickstartAI relevance: {event['why_kickstartai']}

Stance: {event['stance']}

Key arguments:
- {"\n- ".join(event["arguments"])}
""" 
        
        return f"""
{identity_block}

{config_block}

EXAMPLES:
{examples}

EVENT:
{event_block}

TASK:
Write 3 LinkedIn posts based on the EVENT and all constraints above.

Requirements:
- Write AS KickstartAI (first person plural: we, our, us)
- Do NOT describe KickstartAI in third person
- Do NOT mention that you are an AI
- Each post must reflect a different angle:
  1. Strategic (systems, execution, scaling AI)
  2. Reflective (insight, learning, tension)
  3. Opinionated (strong stance, bold perspective)

Content rules:
- Use insights from the EVENT
- Use EXAMPLES only as style reference, not content copying
- Keep tone: confident, collaborative, approachable, human
- Avoid generic AI hype statements
- Focus on real-world application and decisions

Output format:
**Post 1 - Strategic**
...

**Post 2 - Reflective**
...

**Post 3 - Opinionated**
...
"""
    


if __name__ == "__main__":
    # system components
    fs = FewShotPost()
    embedder = Embedder()
    builder = PromptBuilder("config/prompt_config.yaml")

    #1. Fake Interpreter output (structured event)
    interpreter_output = {
        "what_happened": "AI tools are transforming how developers build software",
        "why_global": "AI is reshaping global productivity",
        "why_nl": "Dutch startups are adopting AI rapidly",
        "why_kickstartai": "KickstartAI focuses on AI education and adoption",
        "stance": "AI should be adopted quickly in education",
        "arguments": [
            "Increases productivity",
            "Reduces development time",
            "Creates new opportunities"
        ]
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
    prompt = builder.build(interpreter_output, few_shot_posts)

    print(prompt)


# python3 -m src.Generator.prompt_builder


