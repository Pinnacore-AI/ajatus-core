"""
Ajatuskumppani Personalization Engine
Learns user preferences and adapts AI behavior
"""

import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class UserPreference:
    """User preference structure"""
    user_id: str
    language: str = "fi"
    tone: str = "friendly"  # friendly, professional, casual
    verbosity: str = "medium"  # brief, medium, detailed
    topics_of_interest: List[str] = None
    conversation_style: str = "conversational"  # conversational, direct, educational
    created_at: str = None
    updated_at: str = None
    
    def __post_init__(self):
        if self.topics_of_interest is None:
            self.topics_of_interest = []
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow().isoformat()


@dataclass
class ConversationMemory:
    """Stores conversation history and insights"""
    user_id: str
    conversations: List[Dict] = None
    key_facts: List[str] = None  # Important facts about user
    preferences_learned: Dict = None  # Learned preferences
    interaction_count: int = 0
    last_interaction: str = None
    
    def __post_init__(self):
        if self.conversations is None:
            self.conversations = []
        if self.key_facts is None:
            self.key_facts = []
        if self.preferences_learned is None:
            self.preferences_learned = {}
        if self.last_interaction is None:
            self.last_interaction = datetime.utcnow().isoformat()


class AjatusPersonalizationEngine:
    """Personalization engine for Ajatuskumppani"""
    
    def __init__(self, data_dir: str = "./data/personalization"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.user_preferences: Dict[str, UserPreference] = {}
        self.user_memories: Dict[str, ConversationMemory] = {}
    
    def _get_user_file_path(self, user_id: str, file_type: str) -> str:
        """Get file path for user data"""
        return os.path.join(self.data_dir, f"{user_id}_{file_type}.json")
    
    def load_user_preference(self, user_id: str) -> UserPreference:
        """Load user preference from disk"""
        file_path = self._get_user_file_path(user_id, "preferences")
        
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return UserPreference(**data)
        else:
            # Create new preference
            pref = UserPreference(user_id=user_id)
            self.save_user_preference(pref)
            return pref
    
    def save_user_preference(self, preference: UserPreference):
        """Save user preference to disk"""
        preference.updated_at = datetime.utcnow().isoformat()
        file_path = self._get_user_file_path(preference.user_id, "preferences")
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(asdict(preference), f, ensure_ascii=False, indent=2)
        
        self.user_preferences[preference.user_id] = preference
        logger.info(f"Saved preferences for user {preference.user_id}")
    
    def load_user_memory(self, user_id: str) -> ConversationMemory:
        """Load user conversation memory from disk"""
        file_path = self._get_user_file_path(user_id, "memory")
        
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return ConversationMemory(**data)
        else:
            # Create new memory
            memory = ConversationMemory(user_id=user_id)
            self.save_user_memory(memory)
            return memory
    
    def save_user_memory(self, memory: ConversationMemory):
        """Save user memory to disk"""
        memory.last_interaction = datetime.utcnow().isoformat()
        file_path = self._get_user_file_path(memory.user_id, "memory")
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(asdict(memory), f, ensure_ascii=False, indent=2)
        
        self.user_memories[memory.user_id] = memory
        logger.info(f"Saved memory for user {memory.user_id}")
    
    def add_conversation(
        self,
        user_id: str,
        user_message: str,
        ai_response: str,
        metadata: Optional[Dict] = None
    ):
        """Add conversation to user memory"""
        memory = self.load_user_memory(user_id)
        
        conversation = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_message": user_message,
            "ai_response": ai_response,
            "metadata": metadata or {}
        }
        
        memory.conversations.append(conversation)
        memory.interaction_count += 1
        
        # Keep only last 100 conversations to save space
        if len(memory.conversations) > 100:
            memory.conversations = memory.conversations[-100:]
        
        self.save_user_memory(memory)
    
    def extract_key_facts(self, user_id: str, fact: str):
        """Add key fact about user"""
        memory = self.load_user_memory(user_id)
        
        if fact not in memory.key_facts:
            memory.key_facts.append(fact)
            self.save_user_memory(memory)
            logger.info(f"Added key fact for user {user_id}: {fact}")
    
    def update_preference(
        self,
        user_id: str,
        **kwargs
    ):
        """Update user preference"""
        pref = self.load_user_preference(user_id)
        
        for key, value in kwargs.items():
            if hasattr(pref, key):
                setattr(pref, key, value)
        
        self.save_user_preference(pref)
    
    def get_personalized_system_prompt(self, user_id: str) -> str:
        """Generate personalized system prompt based on user preferences"""
        pref = self.load_user_preference(user_id)
        memory = self.load_user_memory(user_id)
        
        prompt = "Olet Ajatuskumppani, suomalainen avoimen lähdekoodin tekoäly.\n\n"
        
        # Add language preference
        if pref.language == "en":
            prompt = "You are Ajatuskumppani, a Finnish open-source AI.\n\n"
        
        # Add tone
        tone_instructions = {
            "friendly": "Käytä ystävällistä ja lämmintä sävyä.",
            "professional": "Käytä ammattimaista ja asiallista sävyä.",
            "casual": "Käytä rentoa ja epämuodollista sävyä."
        }
        prompt += tone_instructions.get(pref.tone, tone_instructions["friendly"]) + "\n"
        
        # Add verbosity
        verbosity_instructions = {
            "brief": "Pidä vastaukset lyhyinä ja ytimekkäinä.",
            "medium": "Anna tasapainoisia vastauksia.",
            "detailed": "Anna yksityiskohtaisia ja kattavia vastauksia."
        }
        prompt += verbosity_instructions.get(pref.verbosity, verbosity_instructions["medium"]) + "\n"
        
        # Add conversation style
        style_instructions = {
            "conversational": "Keskustele luonnollisesti kuin ystävän kanssa.",
            "direct": "Mene suoraan asiaan ilman turhaa puhetta.",
            "educational": "Selitä asiat opettavaisesti ja yksityiskohtaisesti."
        }
        prompt += style_instructions.get(pref.conversation_style, style_instructions["conversational"]) + "\n"
        
        # Add key facts if available
        if memory.key_facts:
            prompt += "\nTärkeää käyttäjästä:\n"
            for fact in memory.key_facts[-5:]:  # Last 5 facts
                prompt += f"- {fact}\n"
        
        # Add topics of interest
        if pref.topics_of_interest:
            prompt += f"\nKäyttäjä on kiinnostunut: {', '.join(pref.topics_of_interest)}\n"
        
        return prompt
    
    def get_conversation_context(
        self,
        user_id: str,
        last_n: int = 5
    ) -> List[Dict]:
        """Get recent conversation context"""
        memory = self.load_user_memory(user_id)
        
        if not memory.conversations:
            return []
        
        recent = memory.conversations[-last_n:]
        
        context = []
        for conv in recent:
            context.append({"role": "user", "content": conv["user_message"]})
            context.append({"role": "assistant", "content": conv["ai_response"]})
        
        return context
    
    def analyze_user_behavior(self, user_id: str) -> Dict:
        """Analyze user behavior and return insights"""
        memory = self.load_user_memory(user_id)
        pref = self.load_user_preference(user_id)
        
        if not memory.conversations:
            return {"status": "no_data"}
        
        # Calculate average message length
        user_messages = [c["user_message"] for c in memory.conversations]
        avg_length = sum(len(msg) for msg in user_messages) / len(user_messages)
        
        # Detect language
        finnish_words = ["ja", "on", "että", "ei", "ole", "mitä", "mikä"]
        finnish_count = sum(
            any(word in msg.lower() for word in finnish_words)
            for msg in user_messages
        )
        detected_language = "fi" if finnish_count > len(user_messages) / 2 else "en"
        
        return {
            "total_interactions": memory.interaction_count,
            "avg_message_length": avg_length,
            "detected_language": detected_language,
            "key_facts_count": len(memory.key_facts),
            "current_preferences": asdict(pref)
        }


# Example usage
if __name__ == "__main__":
    engine = AjatusPersonalizationEngine()
    
    user_id = "test_user_123"
    
    # Update preferences
    engine.update_preference(
        user_id,
        language="fi",
        tone="friendly",
        verbosity="medium",
        topics_of_interest=["AI", "blockchain", "teknologia"]
    )
    
    # Add conversation
    engine.add_conversation(
        user_id,
        "Hei! Kerro minulle tekoälystä.",
        "Hei! Tekoäly on tietojenkäsittelytieteen ala..."
    )
    
    # Add key fact
    engine.extract_key_facts(user_id, "Käyttäjä on kiinnostunut Web3-teknologiasta")
    
    # Get personalized prompt
    prompt = engine.get_personalized_system_prompt(user_id)
    print("Personalized System Prompt:")
    print(prompt)
    
    # Analyze behavior
    analysis = engine.analyze_user_behavior(user_id)
    print("\nUser Behavior Analysis:")
    print(json.dumps(analysis, indent=2))

