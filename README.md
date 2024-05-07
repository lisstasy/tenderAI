# Tender AI Project

The Tender AI project is a AI-driven solution designed to automate the tendering process by generating commercial offers based on machinery specifications. Here's an overview of the project with technical details:

## Document Processing and Vectorization:
- Utilized LangChain's document loaders to extract technical requirements for machinery from multiple files.
- Employed OpenAI Embeddings for vectorizing the document contents into a database of machinery specifications.

## Semantic Search and Retrieval:
- Leveraged LangChain's Chroma vector store to perform semantic search and retrieval of machinery matching the provided specifications.
- Implemented a retrieval pipeline to efficiently search and retrieve relevant machinery from the vectorized database.

## Natural Language Generation (NLG) for Commercial Offers:
- Developed a natural language generation model using ChatOpenAI with the GPT-4-0125-preview model.
- Integrated the NLG model into the processing chain to generate commercial offers in Russian language, incorporating key data such as capacity, service life, lifting speed, engine type, lifting height, and dimensions.

## Processing Chain and Output Parsing:
- Defined a processing chain consisting of data retrieval, prompt generation, model inference, and output parsing.
- Implemented a ChatPromptTemplate to structure the prompt for model input and output.

## Example Query Execution:
- Provided an example query to demonstrate the system's functionality, instructing the model to generate a commercial offer in Russian based on specified machinery specifications.



