import os
from typing import List

import pinecone
from tqdm.auto import tqdm
from uuid import uuid4
import arxiv

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.vectorstores import Pinecone

INDEX_BATCH_LIMIT = 100

class CharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size, # the character length of the chunk
            chunk_overlap = self.chunk_overlap, # the character length of the overlap between chunks
            length_function = len, # the length function - in this case, character length (aka the python len() fn.)

        )

    def split(self, text: str) -> List[str]:
        return self.text_splitter.split_text(text)

class ArxivLoader:

    def __init__(self, query : str = "Nuclear Fission", max_results : int = 5, encoding: str = "utf-8"):
        """"""
        self.query = query
        self.max_results = max_results
        
        self.paper_urls = []
        self.documents = []
        self.splitter = CharacterTextSplitter()

    def retrieve_urls(self):
        """"""
        arxiv_client = arxiv.Client()
        search = arxiv.Search(
            query = self.query,
            max_results = self.max_results,
            sort_by = arxiv.SortCriterion.Relevance
        )

        for result in arxiv_client.results(search):
            self.paper_urls.append(result.pdf_url)

    def load_documents(self):
        """"""
        for paper_url in self.paper_urls:
            loader = PyPDFLoader(paper_url)
            
            self.documents.append(loader.load())

    def format_document(self, document):
        """"""
        metadata = {
            'source_document' : document.metadata["source"],
            'page_number' : document.metadata["page"]
        }

        record_texts = self.splitter.split(document.page_content)
        record_metadatas = [{
            "chunk": j, "text": text, **metadata
        } for j, text in enumerate(record_texts)]

        return record_texts, record_metadatas
    
    def main(self):
        """"""
        self.retrieve_urls()
        self.load_documents()


class PineconeIndexer:
    
    def __init__(self, index_name : str = "arxiv-paper-index", metric : str = "cosine", n_dims : int = 1536):
        """"""
        pinecone.init(
            api_key=os.environ["PINECONE_API_KEY"],
            environment=os.environ["PINECONE_ENV"]
            )
        
        if index_name not in pinecone.list_indexes():
            # we create a new index
            pinecone.create_index(
                name=index_name,
                metric=metric,
                dimension=n_dims
            )

            self.arxiv_loader = ArxivLoader()
        
        self.index = pinecone.Index(index_name)

    def load_embedder(self):
        """"""
        store = LocalFileStore("./cache/")
        
        core_embeddings_model = OpenAIEmbeddings()

        self.embedder = CacheBackedEmbeddings.from_bytes_store(
            core_embeddings_model,
            store,
            namespace=core_embeddings_model.model
        )

    def upsert(self, texts, metadatas):
        """"""
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeds = self.embedder.embed_documents(texts)
        self.index.upsert(vectors=zip(ids, embeds, metadatas))

    def index_documents(self, documents, batch_limit : int = INDEX_BATCH_LIMIT):
        """"""
        texts = []
        metadatas = []

        # iterate through your top-level document
        for i in tqdm(range(len(documents))):

            # select single document object
            for page in documents[i] : 

                record_texts, record_metadatas = self.arxiv_loader.format_document(page)

                texts.extend(record_texts)
                metadatas.extend(record_metadatas)
            
                if len(texts) >= batch_limit:
                    self.upsert(texts, metadatas)

                    texts = []
                    metadatas = []

        if len(texts) > 0:
            self.upsert(texts, metadatas)

    def get_vectorstore(self):
        """"""
        return Pinecone(self.index, self.embedder.embed_query, "text")


if __name__ == "__main__":
    
    print("-------------- Loading Arxiv --------------")
    axloader = ArxivLoader()
    axloader.retrieve_urls()
    axloader.load_documents()

    print("\n-------------- Splitting sample doc --------------")
    sample_doc = axloader.documents[0]
    sample_page = sample_doc[0]

    splitter = CharacterTextSplitter()
    chunks = splitter.split(sample_page.page_content)
    print(len(chunks))
    print(chunks[0])

    print("\n-------------- testing pinecode indexer --------------")

    pi = PineconeIndexer()
    pi.load_embedder()
    pi.index_documents(axloader.documents)

    print(pi.index.describe_index_stats())
