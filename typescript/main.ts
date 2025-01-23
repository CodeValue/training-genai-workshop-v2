import express from 'express';
import cors from 'cors';
import { v4 as uuidv4 } from 'uuid';

const app = express();
app.use(cors());
app.use(express.json());


app.post('/chat', async (req, res) => {
  const userInput = req.body.content;
  const sessionId = req.body.sessionId || uuidv4(); // Use provided sessionId or generate a new one

  res.status(200).json({ content: userInput, sessionId });
});

app.post('/data', async (req, res) => {
  // Digest data content

  res.sendStatus(200);
});

startServer().catch((error) => {
  console.error('Error starting server', error);
});

async function startServer() {
  app.listen(3000, () => {
    console.log('Server is running on port 3000');
  });
}

async function stopServer() {
  console.log('Server stopped');
}

process.on('SIGINT', async () => {
  console.log('Gracefully shutting down');
  try {
    await stopServer();
    console.log('Database connection closed');
    process.exit(0);
  } catch (error) {
    console.error('Error during shutdown', error);
    process.exit(1);
  }
});
