# AutoGen Chatbot Tutorial – Day 13

This tutorial walks through the basic steps of creating a chatbot using the [AutoGen](https://github.com/microsoft/autogen) framework. The tutorial is broken down into six Python scripts, each building on the previous one. This project is part of a hands-on learning journey and marks **Day 13** of my exploration into AutoGen.

## 📁 Project Files

| **File Name** | **Description** |
|---------------|-----------------|
| `01.py`       | Configuring the Agent System. This script sets up the base configuration required for the chatbot agents to work. |
| `02.py`       | Creating the Agent using the configuration from `01.py`. This script initializes the chatbot agent(s). |
| `03.py`       | Writing a message for the agent and sending it. Demonstrates the communication flow. |
| `04.py`       | Extracting the *required content* from the agent’s reply. Focuses on response parsing and handling. |
| `05.py`       | Taking input from the user and implementing the logic from `04.py`. Makes the bot interactive with user input. |
| `06.py`       | Sending multiple messages in a single-agent system. Introduces batch processing or iterative messaging. |

---

🧠 **Learning Goal:** Understand the core steps involved in configuring, sending, and processing messages between agents using AutoGen.

🛠️ **Tools Required:**  
- Python 3.8+  
- AutoGen (`pip install pyautogen`)  
- OpenAI API key or another LLM provider key  

---

