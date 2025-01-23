// Retrieve or create a sessionId
let sessionId = localStorage.getItem('sessionId');
if (!sessionId) {
  sessionId = crypto.randomUUID();
  localStorage.setItem('sessionId', sessionId);
}

const messagesDiv = document.getElementById('messages');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');

// Utility function to add messages to the chat
function addMessage(content, type) {
  const msgDiv = document.createElement('div');
  msgDiv.classList.add('message', type === 'user' ? 'user-message' : 'ai-message');
  msgDiv.textContent = content;
  messagesDiv.appendChild(msgDiv);
  messagesDiv.scrollTop = messagesDiv.scrollHeight; // Auto-scroll to bottom
}

// Send user message and display AI response
async function sendMessage() {
  const userInput = messageInput.value.trim();
  if (!userInput) return;

  // Display user message
  addMessage(userInput, 'user');
  messageInput.value = '';

  try {
    const response = await fetch('http://localhost:5000/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content: userInput, sessionId })
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const data = await response.json();
    const aiMessage = data.content || 'No response received.';
    addMessage(aiMessage, 'ai');
  } catch (err) {
    console.error(err);
    addMessage('Error: Unable to reach server.', 'ai');
  }
}

// Attach event listeners
sendBtn.addEventListener('click', sendMessage);
messageInput.addEventListener('keyup', (e) => {
  if (e.key === 'Enter') {
    sendMessage();
  }
});
