import os
import re
from functools import cached_property
from itertools import dropwhile
from typing import Iterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class ScriptDocumentLoader(BaseLoader):
    """An "almost" generic loader for a film script file (first use: friends)."""

    def __init__(self, file_path: str, chunk_size: int, overlap_size: int) -> None:
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

    def lazy_load(self) -> Iterator[Document]:
        dialog_lines_chunks = self.__split_with_overlap(
            self.dialog_lines, self.chunk_size, self.overlap_size
        )

        for dialog_lines_chunk in dialog_lines_chunks:
            dialog_chunk = "\n".join(dialog_lines_chunk)
            yield Document(page_content=dialog_chunk, metadata=self.file_metadata)

    @cached_property
    def file_lines(self) -> list[str]:
        try:
            with open(self.file_path, "r", encoding="utf-8") as file_content:
                return [line.rstrip() for line in file_content]
        except Exception as e:
            raise RuntimeError(f"Error loading '{self.file_path}'") from e

    @cached_property
    def dialog_lines(self) -> list[str]:
        compacted_lines = filter(None, self.file_lines)
        without_headers = dropwhile(
            lambda line: not line.startswith("[Scene:"), compacted_lines
        )
        return list(without_headers)

    @cached_property
    def file_metadata(self) -> dict[str, str]:
        metadata = {
            "filename": self.file_path,
            "episode": self.episode_id,
            "title": self.file_lines[0],
            "authors": " ".join(self.file_lines[1:2]).strip(),
        }
        return {k: v for k, v in metadata.items() if v is not None}

    @cached_property
    def episode_id(self) -> str:
        file_name = os.path.basename(self.file_path)
        episode_id_match = re.match(r"^(S[0-9]{2}E[0-9]{2})", file_name)
        return episode_id_match.group() if episode_id_match is not None else None

    def __split_with_overlap(self, lst, chunk_size, overlap_size):
        chunks = []
        for i in range(0, len(lst), chunk_size - overlap_size):
            if i + chunk_size > len(lst):
                break
            chunks.append(lst[i : i + chunk_size])
        return chunks

    @classmethod
    def documents_as_context(cls, documents: Iterator[Document]) -> str:
        """Transform documents into a string for the prompt context."""

        return "\n\n\n".join(cls.document_as_context(document) for document in documents)

    @classmethod
    def document_as_context(cls, document: Document) -> str:
        return f"Dialog in episode {document.metadata['episode']}:\n{document.page_content}"
