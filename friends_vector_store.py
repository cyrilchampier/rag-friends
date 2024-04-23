import os
import shutil
from functools import cached_property
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma

from script_document_loader import ScriptDocumentLoader


class FriendsVectorStore:
    PERSIST_DIRECTORY = "./.chroma_db"
    DOCUMENT_LINES_COUNT = 6
    DOCUMENT_LINES_OVERLAP = 2

    def as_retriever(self, document_retreived_count=20):
        if not os.path.exists(self.PERSIST_DIRECTORY):
            self._create()
        return self.database.as_retriever(search_kwargs={"k": document_retreived_count})

    def refresh(self):
        if os.path.exists(self.PERSIST_DIRECTORY):
            shutil.rmtree(self.PERSIST_DIRECTORY)
        self._create()

    def _create(self):
        print(f"## loading into vector store {len(self.data)}")
        Chroma.from_documents(
            documents=self.data,
            embedding=self.embedding_function,
            persist_directory=self.PERSIST_DIRECTORY,
        )
        print("## finished vector store")

    @cached_property
    def database(self):
        return Chroma(
            persist_directory=self.PERSIST_DIRECTORY,
            embedding_function=self.embedding_function,
        )

    @cached_property
    def embedding_function(self):
        # Same embedding must be used for save and load.
        return GPT4AllEmbeddings()

    @cached_property
    def friends_scripts_path(self):
        # https://www.kaggle.com/datasets/blessondensil294/friends-tv-series-screenplay-script?resource=download
        return (Path.cwd() / "test" / "friends").absolute()

    @cached_property
    def data(self):
        loader = DirectoryLoader(
            self.friends_scripts_path,
            glob="**/*.txt",
            use_multithreading=True,
            show_progress=False,
            loader_cls=ScriptDocumentLoader,
            loader_kwargs={
                "chunk_size": self.DOCUMENT_LINES_COUNT,
                "overlap_size": self.DOCUMENT_LINES_OVERLAP,
            },
        )
        return loader.load()
