from dotenv import load_dotenv
from quart import Quart, request, jsonify
from quart_cors import cors
import uuid

# Load environment variables from the .env file
load_dotenv()

app = cors(Quart(__name__), allow_origin="*") 

@app.route('/chat', methods=['POST'])
async def chat():
    data = await request.get_json()
    user_input = data.get('content')
    session_id = data.get('sessionId') or str(uuid.uuid4())  # Use provided sessionId or generate a new one

    return jsonify({'content': user_input, 'sessionId': session_id})

@app.route('/data', methods=['POST'])
async def digest_content():
    # Digest email content
    
    return '', 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)