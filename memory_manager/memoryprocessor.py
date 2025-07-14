import queue
import threading
import time
import json
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_vertexai import VertexAIEmbeddings
from pathlib import Path
from typing import Dict, Any
import pickle
from langchain_core.messages import BaseMessage

# Memory Management System
class MemorySystem:
    """
    Comprehensive memory system for the voice assistant with:
    - Working Memory: Current conversation context
    - Episodic Memory: Specific past interactions (vector indexed)
    - Semantic Memory: User preferences and facts (vector indexed)
    - Procedural Memory: Interaction patterns and skills (structured data)
    """
    
    def __init__(self, user_id: str, embedding_model: str = "local"):
        self.user_id = user_id
        self.memory_path = Path("memory")
        self.memory_path.mkdir(exist_ok=True)
        
        # Initialize path for user-specific memory
        self.user_memory_path = self.memory_path / user_id
        self.user_memory_path.mkdir(exist_ok=True)
        
        # Set up embeddings model
        if embedding_model == "google":
            try:
                self.embeddings = VertexAIEmbeddings()
                print("Using Google VertexAI embeddings")
            except Exception as e:
                print(f"Error initializing Google embeddings: {e}")
                print("Falling back to local embeddings")
                from langchain_huggingface import HuggingFaceEmbeddings
                self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        else:
            # Use local embedding model by default
            from langchain_huggingface import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            print("Using local HuggingFace embeddings (all-MiniLM-L6-v2)")
            
        # Initialize memory stores
        self._initialize_memory_stores()
        
        # Working memory (current conversation context, limited size)
        self.working_memory = {
            "recent_messages": [],  # Last N messages
            "active_context": {},   # Current context (e.g. active task)
            "attention_focus": None # Current object of attention
        }
        
        # Load saved memories if they exist
        self._load_memories()
        
    def _initialize_memory_stores(self):
        """Initialize the different memory vector stores"""
        # Initialize paths
        (self.user_memory_path / "episodic").mkdir(exist_ok=True)
        (self.user_memory_path / "semantic").mkdir(exist_ok=True)
        (self.user_memory_path / "procedural").mkdir(exist_ok=True)
        
        # Initialize empty vector stores
        self.episodic_memory = FAISS.from_documents(
            [Document(page_content="INITIALIZATION", metadata={"type": "init"})],
            self.embeddings,
        )
        
        self.semantic_memory = FAISS.from_documents(
            [Document(page_content="INITIALIZATION", metadata={"type": "init"})],
            self.embeddings,
        )
        
        # Procedural memory (structured data for learned behaviors)
        self.procedural_memory = {
            "tools": {},              # Tool usage patterns
            "workflows": {},          # Common sequences
            "preferences": {},        # Interaction preferences
            "response_patterns": {},  # Successful response patterns
        }
    
    def _load_memories(self):
        """Load saved memories from disk if they exist"""
        try:
            # Load episodic memory
            episodic_path = self.user_memory_path / "episodic" / "index.faiss"
            if episodic_path.exists():
                self.episodic_memory = FAISS.load_local(
                    self.user_memory_path / "episodic",
                    self.embeddings,
                    "index",
                    allow_dangerous_deserialization=True
                )
            
            # Load semantic memory
            semantic_path = self.user_memory_path / "semantic" / "index.faiss"
            if semantic_path.exists():
                self.semantic_memory = FAISS.load_local(
                    self.user_memory_path / "semantic",
                    self.embeddings,
                    "index",
                    allow_dangerous_deserialization=True
                )
            
            # Load procedural memory
            procedural_path = self.user_memory_path / "procedural" / "data.pkl"
            if procedural_path.exists():
                with open(procedural_path, "rb") as f:
                    self.procedural_memory = pickle.load(f)
                    
            print(f"Loaded memories for user {self.user_id}")
        except Exception as e:
            print(f"Error loading memories: {e}")
            # If loading fails, we'll use the initialized empty memories
    
    def save_memories(self):
        """Save all memories to disk"""
        try:
            # Save episodic memory
            self.episodic_memory.save_local(
                self.user_memory_path / "episodic",
                "index"
            )
            
            # Save semantic memory
            self.semantic_memory.save_local(
                self.user_memory_path / "semantic",
                "index"
            )
            
            # Save procedural memory
            with open(self.user_memory_path / "procedural" / "data.pkl", "wb") as f:
                pickle.dump(self.procedural_memory, f)
                
            #print(f"Saved memories for user {self.user_id}")
        except Exception as e:
            print(f"Error saving memories: {e}")
    
    def update_working_memory(self, messages: List[BaseMessage], context: Dict[str, Any] = None):
        """Update the working memory with recent messages and context"""
        # Keep last 10 messages in working memory
        self.working_memory["recent_messages"] = messages[-10:]
        
        # Update active context if provided
        if context:
            self.working_memory["active_context"].update(context)
    
    def store_episodic_memory(self, interaction: Dict[str, Any]):
        """
        Store a specific interaction in episodic memory
        
        Parameters:
            interaction: Dict with the interaction details including:
                - content: The text content of the interaction
                - timestamp: When it occurred
                - context: Any relevant context
                - metadata: Additional information
        """
        # Create a document with the interaction content
        timestamp = interaction.get("timestamp", datetime.now().isoformat())
        context = interaction.get("context", {})
        metadata = interaction.get("metadata", {})
        
        # Combine relevant information for the document
        content = f"Interaction at {timestamp}: {interaction['content']}"
        
        # Add to episodic memory
        self.episodic_memory.add_documents([
            Document(
                page_content=content,
                metadata={
                    "timestamp": timestamp,
                    "type": "episodic",
                    "context": json.dumps(context),
                    **metadata
                }
            )
        ])
    
    def store_semantic_memory(self, fact: Dict[str, Any]):
        """
        Store a semantic fact or user preference
        
        Parameters:
            fact: Dict with the fact details including:
                - content: The fact text
                - category: Type of fact (preference, knowledge, etc)
                - metadata: Additional information
        """
        category = fact.get("category", "general")
        metadata = fact.get("metadata", {})
        
        # Add to semantic memory
        self.semantic_memory.add_documents([
            Document(
                page_content=fact["content"],
                metadata={
                    "category": category,
                    "type": "semantic",
                    **metadata
                }
            )
        ])
    
    def update_procedural_memory(self, pattern_type: str, pattern_key: str, pattern_data: Any):
        """
        Update procedural memory with new patterns or reinforce existing ones
        
        Parameters:
            pattern_type: Type of pattern (tools, workflows, preferences, response_patterns)
            pattern_key: Identifier for the pattern
            pattern_data: The pattern data to store
        """
        if pattern_type not in self.procedural_memory:
            self.procedural_memory[pattern_type] = {}
            
        # For existing patterns, we might want to update frequency or recency
        if pattern_key in self.procedural_memory[pattern_type]:
            existing_data = self.procedural_memory[pattern_type][pattern_key]
            if isinstance(existing_data, dict) and isinstance(pattern_data, dict):
                # Update the existing data with new values
                existing_data.update(pattern_data)
                # Increment usage count
                existing_data["usage_count"] = existing_data.get("usage_count", 0) + 1
                existing_data["last_used"] = datetime.now().isoformat()
                self.procedural_memory[pattern_type][pattern_key] = existing_data
            else:
                # For simple values, just replace and track usage
                self.procedural_memory[pattern_type][pattern_key] = {
                    "data": pattern_data,
                    "usage_count": 1, 
                    "last_used": datetime.now().isoformat()
                }
        else:
            # For new patterns
            if isinstance(pattern_data, dict):
                pattern_data["usage_count"] = 1
                pattern_data["last_used"] = datetime.now().isoformat()
                self.procedural_memory[pattern_type][pattern_key] = pattern_data
            else:
                self.procedural_memory[pattern_type][pattern_key] = {
                    "data": pattern_data,
                    "usage_count": 1,
                    "last_used": datetime.now().isoformat()
                }
    
    def search_episodic_memory(self, query: str, k: int = 5) -> List[Document]:
        """Search episodic memory for relevant past interactions"""
        return self.episodic_memory.similarity_search(query, k=k)
    
    def search_semantic_memory(self, query: str, k: int = 5, filter_dict: Dict[str, Any] = None) -> List[Document]:
        """Search semantic memory for relevant facts or preferences"""
        if filter_dict:
            # If we have filters, use them
            return self.semantic_memory.similarity_search(
                query, 
                k=k,
                filter=filter_dict
            )
        else:
            return self.semantic_memory.similarity_search(query, k=k)
    
    def search(self, query: str, memory_type: str = "all", k: int = 5) -> List[Dict[str, Any]]:
        """
        Unified search method that searches across memory types
        
        Parameters:
            query: Text to search for
            memory_type: Type of memory to search ("all", "episodic", "semantic", "procedural")
            k: Maximum number of results to return per memory type
            
        Returns:
            List of search results as dictionaries with content, type, and timestamp
        """
        results = []
        
        try:
            # Search episodic memory
            if memory_type in ["all", "episodic"]:
                try:
                    episodic_docs = self.search_episodic_memory(query, k=k)
                    for doc in episodic_docs:
                        results.append({
                            "type": "episodic",
                            "content": doc.page_content,
                            "timestamp": doc.metadata.get("timestamp", "Unknown time")
                        })
                except Exception as e:
                    print(f"Error searching episodic memory: {e}")
            
            # Search semantic memory
            if memory_type in ["all", "semantic"]:
                try:
                    semantic_docs = self.search_semantic_memory(query, k=k)
                    for doc in semantic_docs:
                        results.append({
                            "type": "semantic",
                            "content": doc.page_content,
                            "timestamp": doc.metadata.get("last_updated", "Unknown time")
                        })
                except Exception as e:
                    print(f"Error searching semantic memory: {e}")
            
            # Search procedural memory
            if memory_type in ["all", "procedural"]:
                try:
                    # For each pattern type in procedural memory
                    for pattern_type in ["tools", "workflows", "preferences", "response_patterns"]:
                        patterns = self.get_procedural_pattern(pattern_type, query)
                        for key, value in patterns.items():
                            timestamp = value.get("last_used", "Unknown time") if isinstance(value, dict) else "Unknown time"
                            content = f"{pattern_type.capitalize()}: {key}"
                            results.append({
                                "type": "procedural",
                                "content": content,
                                "timestamp": timestamp
                            })
                except Exception as e:
                    print(f"Error searching procedural memory: {e}")
        
        except Exception as e:
            print(f"Error in memory search: {e}")
        
        # Sort results by relevance (for now, we don't have relevance scores, so just return in order)
        return results
    
    def get_procedural_pattern(self, pattern_type: str, query: str = None) -> Dict[str, Any]:
        """
        Get procedural patterns, optionally filtered by a query
        
        Parameters:
            pattern_type: Type of pattern to retrieve
            query: Optional text to filter patterns
            
        Returns:
            Dictionary of matching patterns
        """
        if pattern_type not in self.procedural_memory:
            return {}
            
        patterns = self.procedural_memory[pattern_type]
        
        # If no query, return the most frequently used patterns
        if not query:
            # Sort by usage count (descending)
            return dict(sorted(
                patterns.items(), 
                key=lambda item: item[1].get("usage_count", 0) 
                if isinstance(item[1], dict) and "usage_count" in item[1] 
                else 0,
                reverse=True
            )[:10])  # Return top 10
        
        # Simple string matching for query
        matched_patterns = {}
        for key, value in patterns.items():
            if query.lower() in key.lower():
                matched_patterns[key] = value
                
        return matched_patterns
    
    def format_for_prompt(self, memory_types: List[str], query: str = None, limit: int = 3) -> str:
        """
        Format relevant memories as context for the LLM prompt
        
        Parameters:
            memory_types: List of memory types to include ("working", "episodic", "semantic", "procedural")
            query: Optional query to search for relevant memories
            limit: Max number of items per memory type
            
        Returns:
            Formatted string with memory context
        """
        memory_context = []
        
        if "working" in memory_types:
            # Add relevant working memory
            recent_msg_summary = "\n".join([
                f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content[:100]}..." 
                for msg in self.working_memory["recent_messages"][-limit:]
            ])
            if recent_msg_summary:
                memory_context.append(f"Recent conversation:\n{recent_msg_summary}")
            
            # Add active context if any
            active_ctx = self.working_memory.get("active_context", {})
            if active_ctx:
                ctx_summary = "\n".join([f"- {k}: {v}" for k, v in active_ctx.items()])
                memory_context.append(f"Current context:\n{ctx_summary}")
        
        if query and "episodic" in memory_types:
            # Add relevant episodic memories
            episodic_results = self.search_episodic_memory(query, k=limit)
            if episodic_results:
                memories = "\n".join([f"- {doc.page_content}" for doc in episodic_results])
                memory_context.append(f"Related past interactions:\n{memories}")
        
        if query and "semantic" in memory_types:
            # Add relevant semantic memories
            semantic_results = self.search_semantic_memory(query, k=limit)
            if semantic_results:
                facts = "\n".join([f"- {doc.page_content}" for doc in semantic_results])
                memory_context.append(f"Relevant facts and preferences:\n{facts}")
        
        if "procedural" in memory_types:
            # Add procedural memory patterns
            if query:
                tool_patterns = self.get_procedural_pattern("tools", query)
                workflow_patterns = self.get_procedural_pattern("workflows", query)
            else:
                tool_patterns = self.get_procedural_pattern("tools")
                workflow_patterns = self.get_procedural_pattern("workflows")
            
            if tool_patterns:
                tools = "\n".join([f"- {k}" for k in list(tool_patterns.keys())[:limit]])
                memory_context.append(f"Relevant tools:\n{tools}")
            
            if workflow_patterns:
                workflows = "\n".join([f"- {k}" for k in list(workflow_patterns.keys())[:limit]])
                memory_context.append(f"Relevant workflows:\n{workflows}")
        
        return "\n\n".join(memory_context)

class BackgroundMemoryProcessor:
    """
    Background processor for memory categorization and storage.
    Uses an LLM to intelligently categorize memories into appropriate types.
    """
    
    def __init__(self, memory_system, llm, max_queue_size=100):
        self.memory_system = memory_system
        self.llm = llm
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.running = False
        self.worker_thread = None
        self.batch_size = 1  # Process one conversation at a time
        self.processing_interval = 5
        
    def start(self):
        """Start the background memory processor"""
        if self.running:
            print("Memory processor already running")
            return
            
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.worker_thread.start()
        print("Background memory processor started")

    def stop(self):
        """Stop the background memory processor"""
        if not self.running:
            return
            
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        print("Background memory processor stopped")
                
    def add_conversation(self, messages):
        """
        Add a conversation to the processing queue
        
        Args:
            messages: List of conversation messages to process for memory extraction
        """
        try:
            # Don't block if queue is full, just log and continue
            if self.queue.full():
                print("Memory processing queue is full, skipping memory extraction")
                return False
                
            # Add to queue
            self.queue.put(messages, block=False)
            return True
        except queue.Full:
            print("Memory processing queue is full, skipping memory extraction")
            return False
    def _process_loop(self):
        """Main processing loop that runs in the background thread"""
        while self.running:
            try:
                # Process a batch of items from the queue
                self._process_batch()
                
                # Sleep between processing batches to avoid high CPU usage
                time.sleep(self.processing_interval)
            except Exception as e:
                print(f"Error in memory processing loop: {e}")
                # Continue processing despite errors
    def _process_batch(self):
        """Process a batch of conversation items from the queue"""
        # Don't process if queue is empty
        if self.queue.empty():
            return
            
        # Process up to batch_size items
        for _ in range(min(self.batch_size, self.queue.qsize())):
            try:
                messages = self.queue.get(block=False)
                self._process_conversation(messages)
                self.queue.task_done()
            except queue.Empty:
                break
    def _process_conversation(self, messages):
        """
        Process a conversation to extract and categorize memories
        
        Args:
            messages: List of conversation messages
        """
        if not messages or len(messages) < 2:
            return
            
        # Get the recent messages for context
        recent_messages = messages[-5:]  # Use last 5 messages for context
        
        # Format conversation for the LLM
        conversation_text = "\n".join([
            f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
            for msg in recent_messages
        ])

        prompt = f"""
        Analyze this conversation and identify information that should be stored in memory.
        
        Conversation:
        {conversation_text}
        
        Extract the following types of information:
        1. Episodic memories (specific events/interactions that happened)
        2. Semantic memories (facts, preferences, knowledge about the user)
        3. Procedural memories (patterns, workflows, or sequences of actions)
        
        Format your response as JSON:
        {{
          "episodic": [
            {{"content": "description of interaction", "importance": 1-10}}
          ],
          "semantic": [
            {{"content": "fact or preference", "category": "preference/knowledge/biographical"}}
          ],
          "procedural": [
            {{"pattern_type": "workflow/tool", "description": "pattern description"}}
          ]
        }}
        
        Rules for good memory extraction:
        - For episodic memories, focus on significant interactions, not routine exchanges
        - For semantic memories, extract actual facts and preferences (things that will be useful later)
        - For procedural memories, identify repeated patterns or workflows the user engages with
        - Assign higher importance (8-10) to memories with personal significance
        - Don't extract memories from hypothetical or speculative statements
        - Don't make up information; only extract what's explicitly in the conversation
        """
        
        try:
            # Get memory classification from LLM
            response = self.llm.invoke(prompt)
            
            # Parse the JSON response
            try:
                memory_data = json.loads(response.content)
                
                # Store each type of memory
                for episodic in memory_data.get("episodic", []):
                    self.memory_system.store_episodic_memory({
                        "content": episodic["content"],
                        "timestamp": datetime.now().isoformat(),
                        "metadata": {"importance": episodic.get("importance", 5)}
                    })
                for semantic in memory_data.get("semantic", []):
                    # Check if similar semantic memory already exists
                    existing_memories = self.memory_system.search_semantic_memory(
                        semantic["content"], 
                        k=3
                    )
                    
                    # If very similar memory exists, skip to avoid duplication
                    if existing_memories and any(
                        self._calculate_similarity(existing.page_content, semantic["content"]) > 0.8
                        for existing in existing_memories
                    ):
                        continue
                        
                    # Otherwise store as new memory
                    self.memory_system.store_semantic_memory({
                        "content": semantic["content"],
                        "category": semantic.get("category", "general")
                    })    
                for procedural in memory_data.get("procedural", []):
                    pattern_type = procedural.get("pattern_type", "workflows")
                    description = procedural.get("description", "")
                    
                    # Create a unique key based on description content
                    pattern_key = f"llm_detected_{hash(description) % 10000}"
                    
                    self.memory_system.update_procedural_memory(
                        pattern_type,
                        pattern_key,
                        {"description": description}
                    )
                
                # Save memories after processing
                self.memory_system.save_memories()
                
            except json.JSONDecodeError:
                print("Error parsing LLM memory classification response as JSON")
                print(f"LLM response: {response.content[:100]}...")
                # Fall back to basic memory extraction
                self._basic_memory_extraction(messages)
        except Exception as e:
            print(f"Error in LLM memory processing: {e}")
            # Fall back to basic memory extraction on error
            self._basic_memory_extraction(messages)
    
    def _calculate_similarity(self, text1, text2):
        """
        Calculate semantic similarity between two texts using embeddings
        """
        # Generate embeddings for both texts
        embedding1 = self.memory_system.embeddings.embed_query(text1)
        embedding2 = self.memory_system.embeddings.embed_query(text2)
        
        # Calculate cosine similarity
        return cosine_similarity([embedding1], [embedding2])[0][0]
    def _basic_memory_extraction(self, messages):
        """
        Fallback method for basic memory extraction when LLM processing fails
        """
        if not messages or len(messages) < 2:
            return
        
        # Get the last user message and assistant response
        user_messages = [m for m in messages[-5:] if isinstance(m, HumanMessage)]
        assistant_messages = [m for m in messages[-5:] if isinstance(m, AIMessage)]
        
        if not user_messages or not assistant_messages:
            return
        
        last_user_msg = user_messages[-1]
        last_assistant_msg = assistant_messages[-1]
        
        # Basic episodic memory extraction
        self.memory_system.store_episodic_memory({
            "content": f"User: {last_user_msg.content} → Assistant: {last_assistant_msg.content[:100]}...",
            "timestamp": datetime.now().isoformat(),
            "context": self.memory_system.working_memory.get("active_context", {})
        })

        user_text = last_user_msg.content.lower()
        if "i like" in user_text or "i prefer" in user_text or "i want" in user_text or "i need" in user_text:
            self.memory_system.store_semantic_memory({
                "content": f"User preference: {last_user_msg.content}",
                "category": "preference"
            })