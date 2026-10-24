from .upload import router as upload_router
from .status import router as status_router
from .result import router as result_router
from .semantic_search import router as semantic_search_router
from .auth import router as auth_router
from .chats import router as chats_router
from .embedding import router as embedding_router
from .categories import router as categories_router
from .tags import router as tags_router
from .vision import router as vision_router
from .layout import router as layout_router
from .translation import router as translation_router

__all__ = [
    'upload_router', 'status_router', 'result_router', 'semantic_search_router',
    'auth_router', 'chats_router', 'embedding_router', 'categories_router',
    'tags_router', 'vision_router', 'layout_router', 'translation_router',
]
