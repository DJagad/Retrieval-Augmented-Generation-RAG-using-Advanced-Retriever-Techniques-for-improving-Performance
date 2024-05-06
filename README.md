# Retrieval Augmented Generation[RAG] using Advanced Retriever Techniques for improving Performance

This repository hosts a Streamlit-based web application designed to demonstrate advanced document retrieval techniques. It leverages the power of OpenAI's embeddings and LangChain's retrieval capabilities to create a versatile tool for querying and extracting information from multiple text sources.

## Features

- **Multiple Retrievers:** Utilizes different retrieval strategies like Parent Document Retriever, Multi-Query Retriever, Contextual Compression Retriever, and Ensemble Retriever.
- **OpenAI Integration:** Incorporates OpenAI's GPT-4 Turbo to enhance query understanding and response generation.
- **User Interface:** Built with Streamlit, providing an intuitive interface for users to interact with the document retrieval system.

## Getting Started

### Prerequisites

- Python 3.8 or newer
- An active OpenAI API key

### Installation

1. **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/document-retrieval-app.git
    cd document-retrieval-app
    ```

2. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Environment Variables**
    Create a `.env` file in the project directory and populate it with your OpenAI API key:
    ```plaintext
    OPENAI_API_KEY='your_openai_api_key_here'
    ```

### Running the Application

To start the Streamlit web application, run the following command in your terminal:
```bash
streamlit run app.py

```
Navigate to http://localhost:8501 in your web browser to view the application.
```

Usage
Select a Retriever: Choose the type of retriever you wish to use from the sidebar.
Input Query: Enter your query in the provided text box.
Submit: Click the "Submit" button to retrieve and display documents related to your query.
Contributing
Contributions to improve the application are welcome. Please fork the repository and submit a pull request with your changes.

License
This project is licensed under the MIT License - see the LICENSE.md file for more details.

css
Copy code

This README file includes basic installation and usage instructions, a brief description, and sections on how to contribute to the project and its licensing information. Adjust the repository URL and any specific instructions according to your actual setup and preferences.