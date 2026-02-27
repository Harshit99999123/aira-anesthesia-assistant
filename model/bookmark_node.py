from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BookmarkNode:
    level: int
    title: str
    start_page: int
    end_page: Optional[int] = None
    children: List["BookmarkNode"] = field(default_factory=list)

    def add_child(self, child: "BookmarkNode"):
        self.children.append(child)

    def __repr__(self):
        indent = "  " * (self.level - 1)
        return f"{indent}Level {self.level}: {self.title} (Pages {self.start_page}-{self.end_page})"