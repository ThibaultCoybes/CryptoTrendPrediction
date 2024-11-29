import express from 'express';
import bodyParser from 'body-parser';
import { TrendPredictionModel, startPrediction } from './main.js';

const app = express();
const port = 3001;

app.use(bodyParser.json());

app.get('/predictions', async (req, res) => {
    try {
        const date = "2024-11-07"
        const crypto = "SOLANA"
        const results = await startPrediction();
        res.status(200).json({ results, crypto, date });  
    } catch (error) {
        console.error('Erreur dans la prédiction:', error);
        res.status(500).send('Erreur lors du traitement de la prédiction');
    }
});

app.listen(port, () => {
    console.log(`API démarrée sur http://localhost:${port}`);
});
