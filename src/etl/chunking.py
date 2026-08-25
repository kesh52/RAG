from abc import ABC, abstractmethod

class BaseChunker(ABC):
    """Abstract base class defining the text chunker interface."""
    
    @abstractmethod
    def split_text(self, text: str) -> list[str]:
        """Split a long text string into smaller, overlap-aware text segments."""
        pass


class RecursiveTextChunker(BaseChunker):
    """Hierarchical text splitter that splits recursively by paragraphs, sentences, and words."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size.")
            
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> list[str]:
        cleaned_text = text.strip()
        if not cleaned_text:
            return []
        return self._split(cleaned_text, self.separators)

    def _split(self, text: str, separators: list[str]) -> list[str]:
        # If the text already fits, return it
        if len(text) <= self.chunk_size:
            return [text]

        # Find the next active separator
        active_separator = ""
        next_separators = []
        for i, sep in enumerate(separators):
            if sep == "" or sep in text:
                active_separator = sep
                next_separators = separators[i+1:]
                break

        # Split the text
        if active_separator == "":
            splits = list(text)
        else:
            splits = text.split(active_separator)

        # Recursively split any parts that are still too large
        final_splits = []
        for part in splits:
            if len(part) > self.chunk_size:
                if next_separators:
                    final_splits.extend(self._split(part, next_separators))
                else:
                    # Hard slice fallback
                    for i in range(0, len(part), self.chunk_size - self.chunk_overlap):
                        final_splits.append(part[i:i+self.chunk_size])
            else:
                final_splits.append(part)

        # Merge splits back together respecting chunk_size and chunk_overlap
        return self._merge_splits(final_splits, active_separator)

    def _merge_splits(self, splits: list[str], separator: str) -> list[str]:
        docs = []
        current_doc = []
        total_len = 0
        
        for s in splits:
            d_len = len(s)
            
            # If adding this split exceeds chunk_size
            sep_len = len(separator) if current_doc else 0
            if current_doc and total_len + d_len + sep_len > self.chunk_size:
                docs.append(separator.join(current_doc))
                
                # Pop from start until remaining fits within chunk_overlap
                while current_doc:
                    popped = current_doc[0]
                    next_len = total_len - len(popped) - (len(separator) if len(current_doc) > 1 else 0)
                    
                    if next_len + d_len + (len(separator) if len(current_doc) > 1 else 0) <= self.chunk_size and next_len <= self.chunk_overlap:
                        current_doc.pop(0)
                        total_len = next_len
                        break
                    else:
                        current_doc.pop(0)
                        total_len = next_len

            current_doc.append(s)
            total_len += d_len + (len(separator) if len(current_doc) > 1 else 0)
            
        if current_doc:
            docs.append(separator.join(current_doc))
            
        return docs
