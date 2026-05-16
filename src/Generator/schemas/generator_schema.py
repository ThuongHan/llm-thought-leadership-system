from pydantic import BaseModel

class LinkedInPosts(BaseModel):
    angle: str
    content: str

class GeneratedPosts(BaseModel):
    posts: list[LinkedInPosts]



    