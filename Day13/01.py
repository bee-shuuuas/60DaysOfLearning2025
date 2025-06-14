from autogen import ConversableAgent
import time 

import os
API_KEY = os.getenv("API_KEY")


print("\n📖 LESSON 1: Configuring the Agent System For First Time")
print("-" * 40)

# =============================================================================
# LESSON 1: Configuring the Agent System
# =============================================================================
 
# Basic Agent Configuration 
basic_config = {
    "config_list" : [{
        "model" : "gemini-1.5-flash",
        "api_key" : API_KEY, 
        "api_type" : "google",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
    }],
    "temperature": 0.7,  # Controls randomness (0 = deterministic, 1 = creative)
    "timeout": 30        # 30 second timeout to avoid hanging
}

print("✅ Agent Configuration Successful using Gemini API Key")
print("-" * 40)