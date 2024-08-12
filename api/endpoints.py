from fastapi import FastAPI
from schemas import Post

#API´s inicializer
api = FastAPI()

#Array de posts
posts = []

#Read root 
@api.get('/')
def read_root():
    return {"Welcome": "Welcome to my API"}

#Get posts
@api.get('/posts')
def get_posts():
    return posts

#Save post
@api.post('/posts')
def save_post(post:Post)->str:
    post.append(post.dict())
    return "recibido"