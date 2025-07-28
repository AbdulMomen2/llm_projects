import psycopg2
from psycopg2.extras import RealDictCursor
import google.generativeai as genai
import json
from datetime import datetime
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
import uuid
import sys

# Load environment variables
load_dotenv()

class ConversationDB:
    def __init__(self, db_config: dict):
        """Initialize database connection and create tables if they don't exist"""
        self.db_config = db_config
        self.test_connection()
        self.create_tables()
    
    def test_connection(self):
        """Test database connection"""
        try:
            conn = psycopg2.connect(**self.db_config)
            conn.close()
            print("✅ Database connection successful!")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            print("Please check your PostgreSQL installation and .env file")
            sys.exit(1)
    
    def get_connection(self):
        """Get database connection"""
        return psycopg2.connect(**self.db_config)
    
    def create_tables(self):
        """Create necessary tables for storing conversations"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Create conversations table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS conversations (
                            id SERIAL PRIMARY KEY,
                            session_id VARCHAR(255) NOT NULL,
                            user_query TEXT NOT NULL,
                            llm_response TEXT NOT NULL,
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            metadata JSONB DEFAULT '{}'
                        );
                    """)
                    
                    # Create indexes for faster queries
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_session_id 
                        ON conversations(session_id);
                    """)
                    
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_timestamp 
                        ON conversations(timestamp);
                    """)
                    
                    conn.commit()
                    print("✅ Database tables created successfully!")
        except Exception as e:
            print(f"❌ Error creating tables: {e}")
            sys.exit(1)
    
    def store_conversation(self, session_id: str, user_query: str, 
                         llm_response: str, metadata: dict = None):
        """Store a conversation record in the database"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO conversations (session_id, user_query, llm_response, metadata)
                        VALUES (%s, %s, %s, %s)
                        RETURNING id;
                    """, (session_id, user_query, llm_response, json.dumps(metadata or {})))
                    
                    record_id = cur.fetchone()[0]
                    conn.commit()
                    return record_id
        except Exception as e:
            print(f"Error storing conversation: {e}")
            return None
    
    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Get conversation history for a session"""
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT * FROM conversations 
                        WHERE session_id = %s 
                        ORDER BY timestamp DESC 
                        LIMIT %s;
                    """, (session_id, limit))
                    
                    return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            print(f"Error getting conversation history: {e}")
            return []
    
    def search_conversations(self, query: str, session_id: str = None) -> List[Dict]:
        """Search conversations by content"""
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    if session_id:
                        cur.execute("""
                            SELECT * FROM conversations 
                            WHERE session_id = %s 
                            AND (user_query ILIKE %s OR llm_response ILIKE %s)
                            ORDER BY timestamp DESC;
                        """, (session_id, f'%{query}%', f'%{query}%'))
                    else:
                        cur.execute("""
                            SELECT * FROM conversations 
                            WHERE user_query ILIKE %s OR llm_response ILIKE %s
                            ORDER BY timestamp DESC;
                        """, (f'%{query}%', f'%{query}%'))
                    
                    return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            print(f"Error searching conversations: {e}")
            return []

class GeminiConversationSystem:
    def __init__(self, gemini_api_key: str, db_config: dict):
        """Initialize Gemini AI and database"""
        try:
            genai.configure(api_key=gemini_api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            print("✅ Gemini API configured successfully!")
        except Exception as e:
            print(f"❌ Error configuring Gemini API: {e}")
            sys.exit(1)
        
        self.db = ConversationDB(db_config)
    
    def chat(self, user_query: str, session_id: str, use_history: bool = True) -> str:
        """Process user query with Gemini and store the conversation"""
        try:
            # Get conversation history if requested
            context = ""
            if use_history:
                history = self.db.get_conversation_history(session_id, limit=5)
                if history:
                    context = "Previous conversation context:\n"
                    for record in reversed(history):  # Reverse to get chronological order
                        context += f"User: {record['user_query']}\n"
                        context += f"Assistant: {record['llm_response']}\n\n"
            
            # Prepare the full prompt
            full_prompt = f"{context}Current question: {user_query}"
            
            # Get response from Gemini
            print("🤖 Generating response...")
            response = self.model.generate_content(full_prompt)
            llm_response = response.text
            
            # Store the conversation in database
            metadata = {
                "model": "gemini-2.5-flash",
                "used_history": use_history,
                "context_length": len(context),
                "timestamp": datetime.now().isoformat()
            }
            
            record_id = self.db.store_conversation(
                session_id=session_id,
                user_query=user_query,
                llm_response=llm_response,
                metadata=metadata
            )
            
            if record_id:
                print(f"💾 Conversation stored with ID: {record_id}")
            
            return llm_response
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def search_previous_conversations(self, search_query: str, session_id: str = None):
        """Search through previous conversations"""
        results = self.db.search_conversations(search_query, session_id)
        return results
    
    def get_session_history(self, session_id: str, limit: int = 10):
        """Get conversation history for a session"""
        return self.db.get_conversation_history(session_id, limit)

def print_separator():
    """Print a nice separator line"""
    print("=" * 60)

def display_conversation_history(history):
    """Display conversation history in a nice format"""
    if not history:
        print("No conversation history found.")
        return
    
    print(f"\n📚 Found {len(history)} conversations:")
    print_separator()
    
    for i, conv in enumerate(reversed(history), 1):
        print(f"[{i}] {conv['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"👤 User: {conv['user_query']}")
        print(f"🤖 AI: {conv['llm_response'][:100]}{'...' if len(conv['llm_response']) > 100 else ''}")
        print("-" * 40)

def main():
    """Main application function"""
    print("🚀 Starting Gemini Conversation Database System")
    print_separator()
    
    # Database configuration from environment variables
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'database': os.getenv('DB_NAME', 'conversation_db'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD'),
        'port': int(os.getenv('DB_PORT', 5432))
    }
    
    # Check if required environment variables are set
    if not os.getenv('GEMINI_API_KEY'):
        print("❌ GEMINI_API_KEY not found in .env file")
        return
    
    if not os.getenv('DB_PASSWORD'):
        print("❌ DB_PASSWORD not found in .env file")
        return
    
    # Initialize the system
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    chat_system = GeminiConversationSystem(gemini_api_key, db_config)
    
    # Generate a session ID for this run
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    print(f"📱 Session ID: {session_id}")
    print_separator()
    
    print("💬 Chat with Gemini! (Type 'quit' to exit, 'history' to see conversation history)")
    print("Commands: 'quit', 'history', 'search <term>', 'new session'")
    print_separator()
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            
            elif user_input.lower() == 'history':
                history = chat_system.get_session_history(session_id)
                display_conversation_history(history)
                continue
            
            elif user_input.lower().startswith('search '):
                search_term = user_input[7:]  # Remove 'search ' prefix
                results = chat_system.search_previous_conversations(search_term, session_id)
                print(f"\n🔍 Search results for '{search_term}':")
                display_conversation_history(results)
                continue
            
            elif user_input.lower() == 'new session':
                session_id = f"session_{uuid.uuid4().hex[:8]}"
                print(f"🆕 Started new session: {session_id}")
                continue
            
            elif not user_input:
                continue
            
            # Get response from Gemini
            response = chat_system.chat(
                user_query=user_input,
                session_id=session_id,
                use_history=True
            )
            
            print(f"\n🤖 Gemini: {response}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()