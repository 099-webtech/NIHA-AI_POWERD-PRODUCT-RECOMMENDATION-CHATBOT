from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import openai
from pymongo import MongoClient
from datetime import datetime
from bson import ObjectId
import json
import traceback
import logging
import os
import speech_recognition as sr  # Add this import for speech recognition
import pinecone
from pinecone import ServerlessSpec
import pandas as pd
import logging
import re

logging.basicConfig(level=logging.INFO)


app = Flask(__name__)
CORS(app, origins=['http://localhost:3000'])  # Adjust the origin as needed

# Create a Pinecone instance
pc = pinecone.Pinecone(
    api_key=''  # Replace with your actual Pinecone API key
)



# Check if the index exists; create it if it doesn't
if 'chat-embeddings-index' not in pc.list_indexes().names():
    pc.create_index(
        name='chat-embeddings-index',  
        dimension=1536,  
        metric='cosine', 
        spec=ServerlessSpec(
            cloud='aws',  
            region='us-east-1'  
        )
    )


# Define the index globally
index = pc.Index('chat-embeddings-index')

# Set your OpenAI API key
OPENAI_API_KEY = ""
openai.api_key = OPENAI_API_KEY  # Set the API key for OpenAI

#  Load CSV file
#csv_file_path = r"E:\Embedded-NiHA\NIHACHAT\NIHA-BOT-backend\flipkart_com-ecommerce_sample.csv"
#df = pd.read_csv(csv_file_path)


# def generate_and_upsert_embeddings(csv_file_path):
#     df = pd.read_csv(csv_file_path)
    
#     for i, row in df.iterrows():
#         text = row['product_name']  # Assuming 'product_name' column contains the text
        
#         try:
#             # Generate the embedding using OpenAI's API
#             embedding_response = openai.Embedding.create(
#                 model='text-embedding-ada-002',  # Using the 'text-embedding-ada-002' model
#                 input=text
#             )
#             embedding = embedding_response['data'][0]['embedding']
            
#             # Prepare a unique ID (e.g., using the index or product name)
#             embedding_id = f'product-{i}'  # Or use 'row["product_name"]' if you prefer to use product name as ID
#              # Prepare metadata by including all columns in the row as metadata
#             metadata = row.to_dict()  # Convert the entire row to a dictionary (each column becomes a key in metadata)
#              # Handle NaN values in metadata: replace them with an empty string or a placeholder like "unknown"
#             metadata = {key: (value if not pd.isna(value) else "unknown") for key, value in metadata.items()}
            
            

#             # Upsert the embedding into Pinecone index
#             index.upsert([(embedding_id, embedding, metadata)])  # Upsert a single vector
#             logging.info(f"Generated embedding for row {i} and upserted to Pinecone.")

#         except Exception as e:
#             logging.error(f"Error generating or upserting embedding for row {i}: {str(e)}")







# MongoDB connection
mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client['chatbot_db']
conversation_collection = db['conversations']

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Update the title generation function (remove async)
def generate_chat_title(messages):
    try:
        title_prompt = {
            "role": "system",
            "content": "Based on the following conversation, generate a short, concise title (max 6 words) that captures the main topic or theme of the discussion."
        }
        
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[title_prompt] + messages,
            max_tokens=10,
            temperature=0.7
        )
        
        return response.choices[0].message['content'].strip()
    except Exception as e:
        logger.error(f"Error generating title: {str(e)}")
        return "Untitled Chat"














@app.route('/chat/<username>', methods=['POST'])
def chat(username):
    data = request.json
    conversation_id = data.get('conversation_id')
    messages = data.get('messages', [])

    if not messages:
        return 'No messages provided', 400

    def generate():
        nonlocal conversation_id, username
        try:
            # Step 1: Use OpenAI to generate response incrementally
            openai_response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                messages=[
                    {"role": "system", "content": (
                        "You are a helpful assistant. If the user input requires product recommendations or cannot be directly "
                        "answered by you, include the phrase 'search_pinecone: true' in your response. Otherwise, provide a normal response."
                    )},
                    *messages
                ],
                stream=True  # Enable streaming
            )

            bot_response = ""
            pinecone_required = False

            # Stream the response chunk by chunk
            for chunk in openai_response:
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    delta = chunk['choices'][0].get('delta', {})
                    content = delta.get('content', "")
                    bot_response += content

                    # Check for Pinecone condition in the streamed response
                    if 'search_pinecone: true' in bot_response.lower():
                        pinecone_required = True
                        # Remove the phrase from the content being sent to the frontend
                        bot_response = bot_response.replace('search_pinecone: true', '').strip()

                    # Send the streamed content to the client
                    yield f"data: {json.dumps({'content': content})}\n\n"

            # If Pinecone search is required, process it incrementally
            if pinecone_required:
                try:
                    user_input_text = messages[-1]['content']  # User's latest input
                    embedding_response = openai.Embedding.create(
                        model='text-embedding-ada-002',
                        input=user_input_text
                    )
                    embedding = embedding_response['data'][0]['embedding']

                    # Query Pinecone for relevant products
                    search_results = index.query(
                        vector=embedding,
                        top_k=1,
                        include_metadata=True
                    )

                    if search_results['matches']:
                        # Stream each match incrementally
                        for match in search_results['matches']:
                            product_metadata = match['metadata']
                            product_name = product_metadata.get('brand', 'No product name')
                            product_price = product_metadata.get('price', 'No price available')
                            category = product_metadata.get('category', 'No category available')
                            product_description = product_metadata.get('description', 'No description available')
                            product_url = product_metadata.get('image', 'No URL available')

                            pinecone_response_chunk = (
                                f"\n\nHere is a relevant product I found:\n\n"
                                f"Name:{product_name}\n"
                                f"Price:{product_price}\n"
                                f"Category:{category}\n"
                                f"Description:{product_description}\n"
                                f"Find more details here:{product_url}"
                            )
                            bot_response += f"\n\n{pinecone_response_chunk}"

                            # Stream Pinecone response chunk to the frontend
                            yield f"data: {json.dumps({'content': pinecone_response_chunk})}\n\n"

                    else:
                        no_match_message = "Sorry, I couldn't find any relevant products based on your query."
                        bot_response += f"\n\n{no_match_message}"
                        yield f"data: {json.dumps({'content': no_match_message})}\n\n"

                except Exception as e:
                    logger.error(f"Error in Pinecone search: {str(e)}")
                    error_message = "Sorry, there was an error processing your request."
                    bot_response += f"\n\n{error_message}"
                    yield f"data: {json.dumps({'content': error_message})}\n\n"

            # Save the entire conversation to the database
            full_conversation = messages + [{'role': 'assistant', 'content': bot_response}]
            if not conversation_id:
                # Create a new conversation if no ID exists
                chat_title = generate_chat_title(full_conversation)
                conversation = {
                    'created_at': datetime.now(),
                    'updated_at': datetime.now(),
                    'messages': full_conversation,
                    'username': username,
                    'title': chat_title
                }
                result = conversation_collection.insert_one(conversation)
                conversation_id = str(result.inserted_id)
                logger.info(f"New conversation created with ID: {conversation_id}")
            else:
                # Update existing conversation
                conversation_collection.update_one(
                    {'_id': ObjectId(conversation_id), 'username': username},
                    {
                        '$set': {
                            'messages': full_conversation,
                            'updated_at': datetime.now(),
                            'username': username
                        }
                    }
                )
                logger.info(f"Conversation updated with ID: {conversation_id}")

            # Stream conversation ID to the frontend
            yield f"data: {json.dumps({'conversation_id': conversation_id})}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            error_details = traceback.format_exc()
            yield f"data: {json.dumps({'error': str(e), 'details': error_details})}\n\n"
            yield "data: [DONE]\n\n"

    return Response(generate(), content_type='text/event-stream')














@app.route('/conversations/<username>', methods=['GET'])
def get_conversations(username):
    try:
        conversations = list(conversation_collection.find(
            {'username': username},
            {'messages': {'$slice': -1}, 'created_at': 1, 'updated_at': 1}
        ))
        for conv in conversations:
            conv['_id'] = str(conv['_id'])
            conv['last_message'] = conv['messages'][0]['content'] if conv.get('messages') else ''
            del conv['messages']
        logger.info(f"Retrieved {len(conversations)} conversations for user {username}")
        return jsonify(conversations)
    except Exception as e:
        logger.error(f"Error retrieving conversations for user {username}: {str(e)}")
        return jsonify({'error': 'Failed to retrieve conversations'}), 500

@app.route('/conversation/<conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    try:
        conversation = conversation_collection.find_one({'_id': ObjectId(conversation_id)})
        if conversation:
            conversation['_id'] = str(conversation['_id'])
            return jsonify(conversation)
        return jsonify({'error': 'Conversation not found'}), 404
    except Exception as e:
        logger.error(f"Error retrieving conversation {conversation_id}: {str(e)}")
        return jsonify({'error': 'Failed to retrieve conversation'}), 500

@app.route('/conversation/<conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id):
    try:
        result = conversation_collection.delete_one({'_id': ObjectId(conversation_id)})
        if result.deleted_count:
            return jsonify({'message': 'Conversation deleted successfully'}), 200
        else:
            return jsonify({'message': 'Conversation not found'}), 404
    except Exception as e:
        logger.error(f"Error deleting conversation {conversation_id}: {str(e)}")
        return jsonify({'error': 'Failed to delete conversation'}), 500

@app.route('/conversations', methods=['POST'])
def create_conversation():
    data = request.json
    username = data.get('username', 'Guest')

    try:
        conversation = {
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
            'messages': [],
            'username': username
        }
        result = conversation_collection.insert_one(conversation)
        conversation_id = str(result.inserted_id)
        logger.info(f"New conversation created with ID: {conversation_id}")
        return jsonify({'conversation_id': conversation_id}), 201
    except Exception as e:
        logger.error(f"Error creating conversation: {str(e)}")
        return jsonify({'error': 'Failed to create conversation'}), 500

@app.route('/chat_sessions/<username>', methods=['GET'])
def get_chat_sessions(username):
    try:
        # Fetch conversations for the specific user
        conversations = list(conversation_collection.find({'username': username}))
        for conv in conversations:
            conv['_id'] = str(conv['_id'])
        return jsonify(conversations)
    except Exception as e:
        logger.error(f"Error retrieving chat sessions for user {username}: {str(e)}")
        return jsonify({'error': 'Failed to retrieve chat sessions'}), 500

@app.route('/groups/<username>', methods=['GET', 'POST'])
def manage_groups(username):
    if request.method == 'GET':
        try:
            groups = list(db.groups.find({'username': username}))
            for group in groups:
                group['_id'] = str(group['_id'])
            return jsonify(groups)
        except Exception as e:
            logger.error(f"Error retrieving groups for user {username}: {str(e)}")
            return jsonify({'error': 'Failed to retrieve groups'}), 500
    
    elif request.method == 'POST':
        data = request.json
        group_name = data.get('name')
        if not group_name:
            return jsonify({'error': 'Group name is required'}), 400
        
        try:
            new_group = {
                'name': group_name,
                'username': username,
                'conversations': [],
                'createdAt': datetime.now()
            }
            result = db.groups.insert_one(new_group)
            new_group['_id'] = str(result.inserted_id)
            return jsonify(new_group), 201
        except Exception as e:
            logger.error(f"Error creating group for user {username}: {str(e)}")
            return jsonify({'error': 'Failed to create group'}), 500

@app.route('/groups/<group_id>/conversations', methods=['POST'])
def add_conversation_to_group(group_id):
    data = request.json
    conversation_id = data.get('conversation_id')
    if not conversation_id:
        return jsonify({'error': 'Conversation ID is required'}), 400
    
    try:
        # Add the conversation to the group
        db.groups.update_one(
            {'_id': ObjectId(group_id)},
            {'$addToSet': {'conversations': ObjectId(conversation_id)}}
        )
        # Optionally, remove the conversation from the main conversations collection
        db.conversations.delete_one({'_id': ObjectId(conversation_id)})
        
        return jsonify({'message': 'Conversation added to group successfully'}), 200
    except Exception as e:
        logger.error(f"Error adding conversation to group {group_id}: {str(e)}")
        return jsonify({'error': 'Failed to add conversation to group'}), 500

@app.route('/groups/<group_id>/conversations', methods=['GET'])
def get_group_conversations(group_id):
    try:
        group = db.groups.find_one({'_id': ObjectId(group_id)})
        if not group:
            return jsonify({'error': 'Group not found'}), 404
        
        conversations = list(conversation_collection.find({'_id': {'$in': group['conversations']}}))
        for conv in conversations:
            conv['_id'] = str(conv['_id'])
        return jsonify(conversations)
    except Exception as e:
        logger.error(f"Error retrieving conversations for group {group_id}: {str(e)}")
        return jsonify({'error': 'Failed to retrieve group conversations'}), 500

@app.route('/groups/<username>', methods=['POST'])
def create_group(username):
    data = request.json
    group_name = data.get('name')
    
    if not group_name:
        return jsonify({'error': 'Group name is required'}), 400

    new_group = {
        'name': group_name,
        'username': username,
        'conversations': [],
        'createdAt': datetime.now()
    }

    try:
        result = db.groups.insert_one(new_group)
        new_group['_id'] = str(result.inserted_id)
        return jsonify(new_group), 201
    except Exception as e:
        logging.error(f"Error creating group: {str(e)}")
        return jsonify({'error': 'Failed to create group'}), 500

@app.route('/groups/<username>/<group_id>', methods=['PUT'])
def rename_group(username, group_id):
    data = request.json
    new_name = data.get('name')
    
    if not new_name:
        return jsonify({'error': 'New group name is required'}), 400

    try:
        result = db.groups.update_one(
            {'_id': ObjectId(group_id), 'username': username},
            {'$set': {'name': new_name}}
        )
        if result.matched_count == 0:
            return jsonify({'error': 'Group not found or does not belong to user'}), 404
        
        return jsonify({'message': 'Group renamed successfully'}), 200
    except Exception as e:
        logger.error(f"Error renaming group {group_id}: {str(e)}")
        return jsonify({'error': 'Failed to rename group'}), 500

@app.route('/groups/<username>/<group_id>', methods=['DELETE'])
def delete_group(username, group_id):
    try:
        result = db.groups.delete_one({'_id': ObjectId(group_id), 'username': username})
        if result.deleted_count == 0:
            return jsonify({'error': 'Group not found or does not belong to user'}), 404
        
        return jsonify({'message': 'Group deleted successfully'}), 200
    except Exception as e:
        logger.error(f"Error deleting group {group_id}: {str(e)}")
        return jsonify({'error': 'Failed to delete group'}), 500

if __name__ == '__main__':
   
    
    # generate_and_upsert_embeddings(csv_file_path)
    app.run(port=2000, debug=True)


   