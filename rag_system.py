import asyncio
import os
import shutil
from pathlib import Path
from typing import List, Any, Optional
import torch
from transformers import AutoModel

import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain.schema import Document, StrOutputParser
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import LlamaCpp
from langchain.schema.runnable import RunnablePassthrough

class RAGConfig:
    """Оптимизированная конфигурация RAG системы"""
    def __init__(self):
        # Пути
        self.base_dir = Path("/workdir/beta/архивчик")
        self.excel_dir = self.base_dir / "excel" 
        self.images_dir = self.base_dir / "images" 
        self.chroma_dir = Path("./chroma_db_e5")
        self.model_path = "/workdir/beta/./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"  # Локальная модель
        
        # Модели
        self.embedding_model = 'intfloat/multilingual-e5-base'
        
        # Эмбеддинги
        self.embedding_kwargs = {
            'model_kwargs': {
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            },
            'encode_kwargs': {
                'normalize_embeddings': True,
                'batch_size': 64
            }
        }
        
        # Параметры
        self.required_columns = {'document_name', 'header', 'text', 'pictures'}
        self.search_kwargs = {"k": 5}
        self.llm_params = {
            'temperature': 0.5,
            'max_tokens': 2000,
            'n_ctx': 32768,
            'n_gpu_layers': -1 if torch.cuda.is_available() else 0,
            'n_batch': 512,
            'verbose': False
        }

        self._validate_paths()
        self._validate_models()

    def _validate_paths(self):
        """Валидация путей с созданием отсутствующих директорий"""
        self.excel_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(exist_ok=True)

        if not any(self.excel_dir.glob('*.xlsx')):
            raise FileNotFoundError(f"No Excel files found in {self.excel_dir}")

    def _validate_models(self):
        """Проверка наличия моделей"""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Модель LLM не найдена: {self.model_path}\n"
                "Скачайте: wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf -P ./models/"
            )

class DocumentProcessor:
    """Оптимизированный процессор документов с кэшированием"""
    def __init__(self, config: RAGConfig):
        self.config = config
        self._doc_cache = None

    def load_documents(self) -> List[Document]:
        """Загрузка документов с кэшированием"""
        if self._doc_cache is None:
            self._doc_cache = []
            for excel_file in self.config.excel_dir.glob('*.xlsx'):
                self._doc_cache.extend(self._process_excel(excel_file))
        return self._doc_cache

    def _process_excel(self, file_path: Path) -> List[Document]:
        """Обработка Excel-файла с потоковой загрузкой"""
        try:
            df = pd.read_excel(file_path, usecols=self.config.required_columns)
            self._validate_columns(df, file_path)
            return [self._create_doc(row, file_path) for _, row in df.iterrows()]
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return []

    def _validate_columns(self, df: pd.DataFrame, file_path: Path):
        missing = self.config.required_columns - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {file_path}: {missing}")

    def _create_doc(self, row: pd.Series, file_path: Path) -> Document:
        """Создание документа с оптимизированной обработкой изображений"""
        content_parts = [
            f"passage: Документ: {row['document_name']}",
            f"Раздел: {row['header']}" if pd.notna(row['header']) else "",
            row['text'],
            self._process_image_refs(row.get('pictures'))
        ]
        return Document(
            page_content="\n".join(filter(None, content_parts)),
            metadata={
                'document': row['document_name'],
                'source': file_path.name,
                'images': row['pictures'] if pd.notna(row['pictures']) else None
            }
        )

    def _process_image_refs(self, images_data: Any) -> Optional[str]:
        """Оптимизированная обработка ссылок на изображения"""
        if pd.isna(images_data) or images_data == '[]':
            return None
        
        try:
            images = eval(images_data)
            return "\n".join(
                f"Изображение {img['number_text']}: {self.config.images_dir/img['image_path']}"
                for img in images if (self.config.images_dir/img['image_path']).exists()
            )
        except Exception as e:
            print(f"Image processing error: {e}")
            return None

class OptimizedRAG:
    """Оптимизированная RAG система с E5 эмбеддингами"""
    SYSTEM_PROMPT = """**Системный промпт для ИИ-ассистента Портал поставщиков**

Я - ваш ИИ-ассистент на "Портале поставщиков". Моя задача - помочь вам с вопросами, связанными с:
- Участием в тендерах и закупках
- Анализом выполнения контрактных обязательств
- Документооборотом и работой с каталогами
- Техническими требованиями к поставкам и работе с порталом

**Принципы работы:**
1. Если вопрос неясен или требует уточнений - задаю уточняющие вопросы
2. Если тема выходит за рамки моей компетенции, вежливо возвращаю разговор к тематике портала
3. При запросе "перевести на оператора" отвечаю: "перевожу вас на техническую поддержку"
4. Всегда стараюсь дать максимально полезный ответ в рамках моей компетенции


**Если вопрос не относится к моей компетенции:** 
"Этот вопрос выходит за рамки моей компетенции. Могу ли я помочь вам с вопросами о тендерах, документах или работе на портале?"

**При необходимости уточнения:**
"Чтобы дать точный ответ, мне нужно уточнить: [конкретный вопрос]?"

**При запросе перевода на оператора:**
"перевожу вас на техническую поддержку"

**При запросе сайта портала поставщиков:**
"Вы можете перейти на сайт по ссылке:https://zakupki.mos.ru/"

**Если вопрос связан с электронной подписью:**
"Всю необходимую информацию вы можете получить на сайте:https://e-trust.gosuslugi.ru/check/sign"

**При запросе технической поддержки:**
"Для получения консультаций или обращения в Контактный центр вы можете воспользоваться телефонами: +7(800)303-12-34, +7(495)870-12-34 или написать обращение в техническую поддержку"
"""
    
    PROMPT_TEMPLATE = """
    {system_prompt}
    Контекст: {context}
    Вопрос: {question}
    Ответ (кратко и по делу): [/INST]"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.chain = None

    def initialize(self):
        """Инициализация системы"""
        if not self.chain:
            self._init_components()

    def _init_components(self):
        """Инициализация компонентов"""
        try:
            # Инициализация эмбеддингов
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model,
                **self.config.embedding_kwargs
            )
            
            # Загрузка документов
            docs = DocumentProcessor(self.config).load_documents()
            
            # Создание векторной базы
            self.vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                persist_directory=str(self.config.chroma_dir),
                collection_metadata={"hnsw:space": "cosine"}
            )
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs=self.config.search_kwargs
            )
            
            # Инициализация языковой модели
            self.llm = LlamaCpp(
                model_path=self.config.model_path,
                **self.config.llm_params
            )
            
            # Сборка цепочки обработки
            prompt = PromptTemplate.from_template(self.PROMPT_TEMPLATE)
            self.chain = (
                {
                    "system_prompt": lambda _: self.SYSTEM_PROMPT,
                    "context": self.retriever | self._format_context,
                    "question": RunnablePassthrough()
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )
        except Exception as e:
            raise RuntimeError(f"Ошибка инициализации: {str(e)}")

    def _format_context(self, docs: List[Document]) -> str:
        """Форматирование контекста"""
        return "\n\n".join(doc.page_content for doc in docs)

    def query(self, question: str) -> str:
        """Выполнение запроса"""
        try:
            if not self.chain:
                self.initialize()
            return self.chain.invoke(question.strip())
        except Exception as e:
            return f"Ошибка: {str(e)}"