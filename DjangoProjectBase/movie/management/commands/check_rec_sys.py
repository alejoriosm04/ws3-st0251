from django.core.management.base import BaseCommand
from movie.models import Movie
import os
import numpy as np

from openai import OpenAI

from dotenv import load_dotenv, find_dotenv

def get_embedding(text, client, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class Command(BaseCommand):
    help = 'Modify path of images'

    def add_arguments(self, parser):
        parser.add_argument('search_term', type=str, help='Search term for recommending movies')

    def handle(self, *args, **options):

        #Se lee del archivo .env la api key de openai
        _ = load_dotenv('../openAI.env')
        client = OpenAI(api_key=os.environ.get('openAI_api_key'))

        # Retrieve the search term from the command arguments
        search_term = options['search_term']

        # Fetch all movies from the database
        items = Movie.objects.all()

        # Get the embedding for the search term
        emb_req = get_embedding(search_term, client)

        # Calculate similarity between the search term embedding and each movie's embedding
        sim = []
        for i in range(len(items)):
            emb = items[i].emb
            emb = list(np.frombuffer(emb))
            sim.append(cosine_similarity(emb,emb_req))
        sim = np.array(sim)

        # Find the index of the highest similarity score
        idx = np.argmax(sim)
        idx = int(idx)

        # Print the title of the most similar movie
        print(items[idx].title)