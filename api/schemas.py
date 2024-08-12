from pydantic import BaseModel
from typing import Text, Optional
from datetime import datetime

#Post Model
class Post(BaseModel):
    id:str
    title:str
    author: Optional[str]
    content:Text
    created_at:datetime=datetime.now()
    publish_at:Optional[datetime]
    published:bool=False