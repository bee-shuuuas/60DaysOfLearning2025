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


# =============================================================================
# LESSON 2: Creating the Agent, using above configuration
# =============================================================================
print("\n📖 LESSON 2: Creating the Agent System")
print("-" * 40)


# Creating a simple agent
my_first_agent = ConversableAgent(
    name="Helper",                              # Give your agent a name
    system_message="You are a helpful tutor.",  # Tell the agent its role
    llm_config=basic_config,                    # Connect to Gemini
    human_input_mode="NEVER",                    # Don't ask for human input
)

print("✅ Created agent named 'Helper'")
print("-" * 40)

# =============================================================================
# LESSON 3.0: Writing a Message for the agents
# =============================================================================

print("-" * 42)

# Step 1: Describe what we're doing
print("\n🤖 Testing basic math")
print("📝 Asking: Explain what 2+2 equals in one sentence.")

# Step 2: Create a message for the agent
messages = [
    {"role": "user", "content": "Explain what 2+2 equals in one sentence."}
]

# =============================================================================
# LESSON 3.1: Sending the Message to the Agent
# =============================================================================

# Step 3: Send the message to the agent and get a reply
reply = my_first_agent.generate_reply(messages=messages)

print("\n Printing the exact message in original format")
print(reply)

# response = str(reply)
# print(response)

# print(type(reply))
# print(type(response))