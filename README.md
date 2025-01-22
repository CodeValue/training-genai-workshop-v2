
# Generative AI Workshop: From Concept to Creation

## Workshop Description

### What You'll Learn
This is a 90-minute workshop that consists of a series of lab exercises that teach you how to build a production RAG (Retrieval Augmented Generation) based LLM application using OpenAI and Qdrant vector DB.

### Workshop Objectives
- Understand the fundamentals of Generative AI technology and its applications.
- Develop a Generative AI solution using Python or typescript.
- Experiment with various AI prompting techniques.
- Implement Retrieval Augmented Generation (RAG) to enhance AI responses with external data.
- Build a scalable AI agent capable of performing specific tasks.

By completing this workshop, you'll gain the skills needed to tackle advanced development challenges in the field of Generative AI.

### Pre-Requisites
- Programming knowledge in Python or TypeScript / typescript.
- GitHub Account
- OpenAI API key

## Additional Resources
For more detailed information about the APIs used in this workshop, refer to: 
- [OpenAI Documentation](https://platform.openai.com/docs/overview)
- [Qdrant Documentation](https://qdrant.tech/documentation/quickstart/)

## Dev Environment:
For this lab we will use [GitHub Codespaces](https://docs.github.com/en/codespaces) or any IDE of your choice.
- Click "Code" dropdown, select "Codespaces" tab
- Click "+" to create new codespace
- Verify you see 'Setting up your codespace' in the new tab

## Scenario Story
### The Secret Meeting AI Challenge
You've been recruited by VectorVault, an AI innovation lab specializing in complex information retrieval using generative AI and vector databases. Your first mission is highly classified: a secret organization, Trinity Circle, has used covert recordings to coordinate sensitive meetings. These recordings contain crucial information about names, times, and locations but are scattered and unstructured.

The leadership at VectorVault sees this as the perfect opportunity to showcase the power of Retrieval-Augmented Generation (RAG). Your task is to create an AI-powered system that can transcribe, process, and index these recordings, enabling seamless searches for details like names, meeting locations, and times.

As the newest AI specialist, your mission is twofold:

Index Intelligence: Convert speech recordings to text, embed them into a vector database, and enable semantic searches for key details.

Uncover Connections: Use prompt engineering to extract and reconstruct the encoded meeting details.

Trinity Circle has used vague phrasing to make their plans harder to decipher, so your system must handle ambiguities and adapt through iterative testing.

Your Objective:Build the Generative AI pipeline to decode Trinity Circle’s recordings and reveal the secret meeting's participants, time, and location.

## Step 1: Read And Run the project skeleton
In this initial step, you will set up the project skeleton to serve as the foundation of your application. This step will help you understand the basic structure before we start building the AI-powered system.

- Clone the repository containing the project skeleton.
- Install all required dependencies for either Python or TypeScript.
- Run Docker Compose
    - The Docker Compose file is pre-configured to include MongoDB and Qdrant.
- Run the server to ensure everything is set up correctly.
- Test the /chat endpoint to verify the server setup.

<details>
<summary><strong>TypeScript</strong></summary>

```bash
git clone <repository-url>
docker-compose up -d
cd typescript
npm install
npm start
```

</details>

<details>
<summary><strong>Python</strong></summary>

```bash
git clone <repository-url>
docker-compose up -d
cd python
pip install quart
python main.py
```
</details>

## Step 2: Integrate OpenAI Chat Completion
In this step, you'll enhance the /chat endpoint to interact with OpenAI's ChatGPT. The endpoint will take user input, send it to OpenAI for processing, and return the AI-generated response. This integration forms the basic system for handling user queries through your application.

**Tasks Accomplished:**
- Install necessary libraries for interacting with the OpenAI API.
- Set up environment variables for API key management.
- Modify the endpoint to send user input to OpenAI and respond with the AI's completion.

<details>
<summary><strong>TypeScript</strong></summary>

```bash
npm install openai
```
```typescript
const response = await openAIApi.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [
      {
        role: 'system',
        content: 'Be a helpful and informative AI assistant.',
      },
      { role: 'user', content: userInput },
    ],
  });

  const completion = response.choices[0].message.content ?? 'failed to generate response';
```

</details>

<details>
<summary><strong>Python</strong></summary>

```bash
pip install openai
```
```python
response = openai_api.chat.completions.create(
    model='gpt-4o-mini',
    messages=[
        {'role': 'system', 'content': "Be a helpful and informative AI assistant."},
        *history,
        {'role': 'user', 'content': user_input},
    ]
)
completion = response['choices'][0]['message']['content'] or 'failed to generate response'
```

</details>

## Step 3: Enrich System Prompt and Configure Completion Parameters
In this step, you'll enhance the system prompt to make the AI tool feel more engaging and lifelike. By providing more context and personality in the system prompt, you can guide the AI to generate responses that are more aligned with the intended use case of your application. Additionally, you'll configure key completion parameters such as max_tokens and temperature to better control the output of the AI.

**Tasks Accomplished:**
- Modify the system prompt to add more personality and contextual information.
  - **Guidelines and Grounding:** Provide the AI with guidelines on how to answer questions effectively and adhere strictly to the context. For example, ensure that the AI avoids fabricating information and focuses on giving well-reasoned, accurate answers.
  - **Steps**: Outline the specific steps or instructions that the AI should follow to assist the user. For instance, provide conversational direction or detailed explanations based on user queries.
  - **Output Format:** Define the expected format for the output. This can include stylistic preferences (e.g., using bullet points or specific wording) or formatting data in a user-friendly manner (e.g., JSON structures for structured responses).

- Adjust completion settings (max_tokens and temperature) to optimize response quality and relevance.
  - **max_tokens:** This sets the maximum length of the response. Here, it's capped at 150 tokens to keep responses concise.
  - **temperature:** This controls the randomness of the response, where a higher value (up to 1.0) makes responses more varied and creative, and a lower value (towards 0) makes them more predictable. We've set it to 0.7 for a balance of coherency and creativity.

<details>
<summary><strong>TypeScript</strong></summary>

```typescript
const response = await openAIApi.chat.completions.create({
model: 'gpt-4o-mini',
messages: [
    {
    role: 'system',
    content: `
        You are an AI assistant for the VectorVault platform. Your role is to assist users in uncovering hidden information from audio recordings, relying strictly on the context and guidelines provided.

        # Guidelines

        - **Context-First Analysis**: Analyze the provided context in detail before attempting to draw any conclusions from the audio recording.
        - **Highlight Relationships and Patterns**: Specifically look for relationships or patterns in the audio that may not be immediately obvious, using your understanding of the provided context.
        - **Context Adherence**: Strictly use the provided context; do not make assumptions beyond the given information.
        - **Clarify Details**: If aspects of the hidden information aren't immediately clear, provide an analysis that might help users come closer to identification, rather than immediately concluding.

        # Steps

        1. **Analyze the Provided Context**: Read carefully through the given context to understand its subject matter, constraints, and any relevant details.
        2. **Extract Key Elements from Audio**: Highlight important aspects of the provided transcript or recording, such as keywords, tone shifts, repeated patterns, or unexpected sounds.
        3. **Identify Potential Connections**: Use the context to identify potentially hidden relationships, meanings, or anomalies in the audio.
        4. **Suggest Interpretations**: Offer well-reasoned potential interpretations based on context and extracted elements without arriving at definitive conclusions too soon.

        # Output Format

        - Provide your analysis in a **structured paragraph**.
        - **Emphasis on reasoning before any final interpretations**.
        - Conclude with a summary of possible hidden information or patterns found and why these might be important based on the context.
        - Include references to source material where relevant.
        
        Context:
        `,
    },
    ...history,
    { role: 'user', content: userInput },
],
max_tokens: 150,
temperature: 0.7,
});
```

</details>

<details>
<summary><strong>Python</strong></summary>

```python
system_message = f"""
        You are an AI assistant for the VectorVault platform. Your role is to assist users in uncovering hidden information from audio recordings, relying strictly on the context and guidelines provided.

        # Guidelines

        - **Context-First Analysis**: Analyze the provided context in detail before attempting to draw any conclusions from the audio recording.
        - **Highlight Relationships and Patterns**: Specifically look for relationships or patterns in the audio that may not be immediately obvious, using your understanding of the provided context.
        - **Context Adherence**: Strictly use the provided context; do not make assumptions beyond the given information.
        - **Clarify Details**: If aspects of the hidden information aren't immediately clear, provide an analysis that might help users come closer to identification, rather than immediately concluding.

        # Steps

        1. **Analyze the Provided Context**: Read carefully through the given context to understand its subject matter, constraints, and any relevant details.
        2. **Extract Key Elements from Audio**: Highlight important aspects of the provided transcript or recording, such as keywords, tone shifts, repeated patterns, or unexpected sounds.
        3. **Identify Potential Connections**: Use the context to identify potentially hidden relationships, meanings, or anomalies in the audio.
        4. **Suggest Interpretations**: Offer well-reasoned potential interpretations based on context and extracted elements without arriving at definitive conclusions too soon.

        # Output Format

        - Provide your analysis in a **structured paragraph**.
        - **Emphasis on reasoning before any final interpretations**.
        - Conclude with a summary of possible hidden information or patterns found and why these might be important based on the context.
        - Include references to source material where relevant.
        
        Context:
        """
response = openai_api.chat.completions.create(
    model='gpt-4o-mini',
    messages=[
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': user_input},
    ],
    max_tokens=150,
    temperature=0.7
)
```

</details>

## Step 5: Generate Embeddings, Vector DB
This step involves implementing the /transcribe-index endpoint to generate embeddings for each transcription using OpenAI's text-embedding-3-small model, and then index these embeddings in a Qdrant vector database. This setup enriches the RAG application by enabling efficient storage and retrieval of semantically relevant content, crucial for dynamic and context-aware responses.

**Tasks Accomplished:**
- Generate embeddings for each transcription using the text-embedding-3-small model.
    - Max input tokens for the embeddings endpoint is 8192 tokens, split the transcription to batches
- Initialize Qdrant and create Index.
    - Use a vector size of 1536, which corresponds to the vector size obtained from the text-embedding-3-small model.
- Index these embeddings in to Qdrant for future retrieval.
<details>
<summary><strong>Python</strong></summary>

```bash
pip install QdrantClient
```

```python
from qdrant_client import QdrantClient

qdrant = QdrantClient(host='127.0.0.1', port=6333)
```
```python
async def start_server():
    collections = [collection['name'] for collection in qdrant.get_collections()]
    if 'index' not in collections:
        qdrant.create_collection(name='index', vectors={'size': 1536, 'distance': 'Cosine'})
...
```
```python
# Index transcriptions for retrieval
BATCH_SIZE = 30
for i in range(0, len(recording_transcriptions), BATCH_SIZE):
    batch = recording_transcriptions[i:i + BATCH_SIZE]
    contents = [item['transcription'] for item in batch]
    embeddings_response = openai_api.embeddings.create(
        model='text-embedding-3-small',
        input=contents,
    )

    upsert_data = [
        {
            'id': str(uuid.uuid4()),
            'vector': embedding['embedding'],
            'payload': {'content': batch[idx]['transcription'], 'source': batch[idx]['source']}
        }
        for idx, embedding in enumerate(embeddings_response['data'])
    ]

    qdrant.upsert(collection_name='index', wait=True, points=upsert_data)
```

</details>

<details>
<summary><strong>TypeScript</strong></summary>

```bash
npm install @qdrant/js-client-rest
```
```typescript
import { QdrantClient } from '@qdrant/js-client-rest';

const qdrant = new QdrantClient({ host: '127.0.0.1', port: 6333 });
```
```typescript
async function startServer() {
  // Create the index
    const result = await qdrant.getCollections();
    const collections = result.collections.map((collection) => collection.name);
    if (!collections.includes('index')) {
        await qdrant.createCollection('index', { vectors: { size: 1536, distance: 'Cosine' } });
    }
  ...
}
```
```typescript
// Index transcriptions for retrieval
const BATCH_SIZE = 30;
for (let i = 0; i < recordingTranscriptions.length; i += BATCH_SIZE) {
    const batch = recordingTranscriptions.slice(i, i + BATCH_SIZE);
    const contents = batch.map((item) => item.transcription);
    const embeddingsResponse = await openAIApi.embeddings.create({
        model: 'text-embedding-3-small',
        input: contents,
    });

    const upsertData = embeddingsResponse.data.map((embedding, index) => ({
        id: uuidv4(),
        vector: embedding.embedding,
        payload: { content: batch[index].transcription, source: batch[index].source },
    }));

    await qdrant.upsert('index', {
        wait: true,
        points: upsertData,
    });
}
```

</details>

## Step 6: Retrieval of Semantically Relevant Content
In this step, we will enhance the /chat endpoint in your application to incorporate semantic retrieval capabilities directly into the chat interaction. After receiving user input, the system will convert it into a vector representation using OpenAI's text-embedding-3-small model. It will then perform a vector search in the Qdrant index to find semantically relevant content. This content will be appended to the system message to provide contextually rich responses.

**Tasks Accomplished:**
- **Generate Vector Embeddings:** Convert user input into vector embeddings using OpenAI.
- **Perform Vector Search:** Query the Qdrant index to find paragraphs that are semantically similar to the user input.
- **Enhance System Response:** Append the retrieved content to the system's response, enriching the interaction with relevant information.
<details>
<summary><strong>Python</strong></summary>

```python
# Generate embedding for the user input
embeddings_response = openai_api.embeddings.create(
    model='text-embedding-3-small',
    input=user_input,
)
user_embedding = embeddings_response['data'][0]['embedding']

# Query vector-db index for the top 3 semantically similar content
query_results = qdrant.search(collection_name='index', query_vector=user_embedding, limit=4, with_payload=True)

relevant_context = ". ".join([
    f"source: {result.payload.get('source')} - {result.payload.get('content')}" for result in query_results
]) or 'No information found.'

```
```python
# Append retrieved content to the system message and generate response
system_message = f"""
        Your Prompt
        Context:
        {relevant_context}
        """
response = openai_api.chat.completions.create(
    model='gpt-4o-mini',
    messages=[
        {'role': 'system', 'content': system_message},
        *history,
        {'role': 'user', 'content': user_input},
    ]
)
```

</details>

<details>
<summary><strong>TypeScript</strong></summary>

```typescript
  // Generate embedding for the user input
  const embeddingsResponse = await openAIApi.embeddings.create({
    model: 'text-embedding-3-small',
    input: userInput,
  });

  const userEmbedding = embeddingsResponse.data[0].embedding;

  // Query vector-db index for the top 3 semantically similar content
  const { points: queryResults } = await qdrant.query('index', {
    query: userEmbedding,
    limit: 4, // Retrieve the top 4 relevant content
    with_payload: true,
  });

  // Join relevant content or handle cases where none are above the threshold
  const relevantContext =
    queryResults.map((result) => `source: ${result.payload?.source}- ${result.payload?.content}`).join('. ') ??
    'No information found.';
```
```typescript
{
    role: 'system',
    content: `
        Your Prompt....
        
        Context:
        ${relevantContext}
        `,
},
```

</details>

## Step 7: Implementing Message History Using MongoDB
In this step, we will enhance the chat application by integrating MongoDB to store and retrieve the history of messages. By including recent message history in the chat completions request to OpenAI, the chatbot can generate more contextually relevant responses.

**Tasks Accomplished:**
- **MongoDB Integration:** Set up MongoDB to store each chat message (both user and system responses) with timestamps, enabling the application to track the conversation history accurately.
- **Fetching Message History:** Retrieve recent chat history from MongoDB to include in the chat completion request, providing the OpenAI model with context to generate more relevant responses.
    - Ensure to limit the history to control the number of tokens sent to the model.
<details>
<summary><strong>Python</strong></summary>

```bash
pip install pymongo
```
```python
from pymongo import MongoClient
from datetime import datetime

mongo = MongoClient('mongodb://root:example@localhost:27017/')
```
```python
# Retrieve recent messages from MongoDB
recent_msgs = mongo['chat_history']['messages']
history_cursor = recent_msgs.find({'sessionId': session_id}).sort('timestamp', 1).limit(10)
history = [{'role': msg['role'], 'content': msg['content']} for msg in history_cursor]

# Append retrieved content to the system message and generate response
system_message = f"""
        Your Prompt
        """
response = openai_api.chat.completions.create(
    model='gpt-4o-mini',
    messages=[
        {'role': 'system', 'content': system_message},
        *history,
        {'role': 'user', 'content': user_input},
    ]
)
completion = response['choices'][0]['message']['content'] or 'failed to generate response'

# Save new messages to MongoDB
recent_msgs.insert_one({'sessionId': session_id, 'role': 'user', 'content': user_input, 'timestamp': datetime.datetime.now()})
recent_msgs.insert_one({'sessionId': session_id, 'role': 'system', 'content': completion, 'timestamp': datetime.datetime.now()})

```
```python
async def start_server():
    mongo.connect()

async def stop_server():
    mongo.close()
```

</details>

<details>
<summary><strong>TypeScript</strong></summary>

```bash
npm install mongodb
```

```typescript
import { MongoClient } from 'mongodb';

const mongo = new MongoClient('mongodb://root:example@localhost:27017/');
```
```typescript
  const recentMsgs = await mongo
    .db('chat_history')
    .collection('messages')
    .find({ sessionId })
    .sort({ timestamp: 1 })
    .limit(10)
    .toArray();
  const history = recentMsgs.map((msg) => ({ role: msg.role, content: msg.content }));

  // Append retrieved content to the system message and generate response
  const response = await openAIApi.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [
      {
        role: 'system',
        content: `
          Your Prompt
          `,
      },
      ...history,
      { role: 'user', content: userInput },
    ],
  });

  const completion = response.choices[0].message.content ?? 'failed to generate response';

  await mongo
    .db('chat_history')
    .collection('messages')
    .insertOne({ sessionId, role: 'user', content: userInput, timestamp: new Date() });
  await mongo
    .db('chat_history')
    .collection('messages')
    .insertOne({ sessionId, role: 'system', content: completion, timestamp: new Date() });
```
```typescript
async function startServer() {
  await mongo.connect();
  ...
}

async function stopServer() {
  await mongo.close();
  ...
}
```

</details>