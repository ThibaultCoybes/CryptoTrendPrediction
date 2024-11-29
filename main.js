import fetch from 'node-fetch';
import tf from '@tensorflow/tfjs-node';
import fs from 'fs';
import path from 'path';
import {dirname} from 'path';
import { fileURLToPath } from 'url';

const articlesFilePath = './dogecoin_articles_gain.json'; 
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const modelSavePath = path.join(__dirname, 'model');
const API_KEY = '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';
const API_URL = 'https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment';

if (!fs.existsSync(modelSavePath)) fs.mkdirSync(modelSavePath);

export const loadArticles = () => {
    try {
        const data = fs.readFileSync(articlesFilePath, 'utf8');
        return JSON.parse(data);
    } catch (error) {
        console.error('Erreur lors du chargement des articles:', error);
        return [];
    }
};

export class TrendPredictionModel {
    constructor() {
        this.experienceReplayMemory = [];
        this.points = 0;
        this.consecutiveErrors = 0;
        console.log('Modèle initialisé avec', this.points, 'points.');
    }

    createModel() {
        const model = tf.sequential();
        model.add(tf.layers.dense({ units: 8, activation: 'relu', inputShape: [1] }));
        model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));
        model.compile({ optimizer: tf.train.adam(0.0001), loss: 'sparseCategoricalCrossentropy', metrics: ['accuracy'] });
        return model;
    }

    async saveModel() {
        try {
            await this.model.save(`file://${modelSavePath}`);
            console.log("Modèle sauvegardé.");
        } catch (error) {
            console.error("Erreur lors de la sauvegarde du modèle :", error);
        }
    }
    
    async trainModel(states, targetQValues) {
        const input = tf.tensor2d(states, [states.length, 1], 'float32');
        const targetTensor = tf.tensor2d(targetQValues, [targetQValues.length, 1], 'float32'); 
        await this.model.fit(input, targetTensor, {
            epochs: 40,
            batchSize: 16,
            shuffle: true,
            callbacks: [tf.callbacks.earlyStopping({ monitor: 'loss', patience: 2 })],
        });
        await this.saveModel();
    }
    
    async loadModel() {
        const modelPath = path.join(modelSavePath, "/model.json");
        if (fs.existsSync(modelPath)) {
            try {
                this.model = await tf.loadLayersModel(`file://${modelPath}`);
                console.log("Modèle chargé.");
    
                if (!this.model.optimizer) {
                    this.model.compile({
                        optimizer: tf.train.adam(0.0001),
                        loss: 'sparseCategoricalCrossentropy',
                        metrics: ['accuracy']
                    });
                    console.log("Modèle compilé.");
                }
            } catch (error) {
                console.error("Erreur lors du chargement du modèle :", error);
                this.model = this.createModel(); 
            }
        } else {
            console.log("Aucun modèle trouvé, création d'un nouveau modèle.");
            this.model = this.createModel(); 
        }
    }

    async analyzeSentiment(articles) {
        let totalSentiment = 0;
        for (let article of articles) {
            const sentimentScore = await this.analyzeArticleSentiment(article.content);
            totalSentiment += sentimentScore;
        }
        return totalSentiment / articles.length;
    }

    async analyzeArticleSentiment(content) {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${API_KEY}`, 'Content-Type': 'application/json' },
            body: JSON.stringify({ inputs: content }),
        });

        if (!response.ok) {
            console.error(`Erreur API pour l'article : "${content}"`);
            throw new Error('Erreur API Hugging Face');
        }

        const result = await response.json();
        const bestSentiment = result[0].reduce((prev, curr) => (curr.score > prev.score ? curr : prev));
        return bestSentiment.label === 'LABEL_0' ? -1 : bestSentiment.label === 'LABEL_2' ? 1 : 0;
    }

    async decideTrend(globalSentiment) {
        const inputTensor = tf.tensor2d([globalSentiment], [1, 1], 'float32');
        let prediction = this.model.predict(inputTensor).dataSync();
    
        return tf.tensor1d(prediction).argMax().dataSync()[0];
    }
    
    
    adjustSentimentDynamically(currentSentiment) {
        const variation = (Math.random() - 0.5) * 0.1;  // Variation légère
        let newSentiment = currentSentiment + variation;
    
        // Limiter le sentiment entre 0 et 1 pour éviter les valeurs non valides
        newSentiment = Math.max(0, Math.min(1, newSentiment));
    
        return newSentiment;
    }
    
    async predictTrend(articles) {
        let globalSentiment = await this.analyzeSentiment(articles);
        globalSentiment = this.adjustSentimentDynamically(globalSentiment);
        const decision = await this.decideTrend(globalSentiment);
        return { globalSentiment, decision };
    }

    addExperience(state, action, reward, nextState) {
        const isRepeatError = this.experienceReplayMemory.some(
            exp => exp.state === state && exp.action === action && reward < 0
        );
    
        let adjustedReward = reward;
    
        if (isRepeatError) {
            // Pénalité pour erreurs répétées
            adjustedReward *= 1.5;
        } else if (this.consecutiveErrors >= 3 && reward < 0) {
            // Récompense pour essayer une nouvelle action après plusieurs erreurs
            adjustedReward += 1.5;
        }
    
        this.experienceReplayMemory.push({ state, action, reward: adjustedReward, nextState });
    
        if (this.experienceReplayMemory.length > 1000) this.experienceReplayMemory.shift();
        this.points += adjustedReward;
    }
    

    async learnFromExperience() {
        if (this.experienceReplayMemory.length < 2) return;
    
        const batch = this.experienceReplayMemory.slice(-2);
    
        const states = batch.map(exp => exp.state);
        const actions = batch.map(exp => exp.action);
        const rewards = batch.map(exp => exp.reward);
        const nextStates = batch.map(exp => exp.nextState);
    
        const nextStateTensor = tf.tensor2d(nextStates, [nextStates.length, 1], 'float32');
        
        const nextQValues = this.model.predict(nextStateTensor).arraySync();
    
        const maxNextQValues = nextQValues.map(q => Math.max(...q));
        const updatedQValues = rewards.map((reward, i) => reward + 0.9 * maxNextQValues[i]);
    
        const targetQValues = updatedQValues.map((value, index) => actions[index]);

        await this.trainModel(states, targetQValues);
    }
    

    async checkPrediction(articles, realTrend) {
        console.log("Real Trend", realTrend)
        const { globalSentiment, decision } = await this.predictTrend(articles);
        this.logPrediction(globalSentiment, decision, realTrend);
    
        const reward = (decision === realTrend) ? 5 : -1;
        if (this.consecutiveErrors >= 3) {
            console.log("Bonus pour l'effort après plusieurs erreurs consécutives.");
            reward += 2; // Donnez un petit bonus
        }
        this.addExperience(globalSentiment, decision, reward, realTrend);
        console.log("decision : ", decision)
        if (this.experienceReplayMemory.length >= 2) {
            await this.learnFromExperience(); 
        }
        return decision
    }

    logPrediction(globalSentiment, predictedTrend, realTrend) {
        const isCorrect = predictedTrend === realTrend;
        const errorAnalysis = isCorrect ? '' : `Erreur détectée : Sentiment=${globalSentiment}, Prédit=${predictedTrend}, Réel=${realTrend}`;
        const logMessage = `[${new Date().toISOString()}] ${errorAnalysis} Points: ${this.points}\n`;
        fs.appendFileSync('predictions_log.txt', logMessage);
    
        if (!isCorrect) {
            console.log("Analyse de l'erreur :", errorAnalysis);
            this.adjustModelForError(globalSentiment, predictedTrend, realTrend);
        }
    }
    adjustModelForError(globalSentiment, predictedTrend, realTrend) {
        // Ajouter une pénalité spécifique au modèle
        this.addExperience(globalSentiment, predictedTrend, -3, realTrend);
    }
}

export const startPrediction = async () => {
    const model = new TrendPredictionModel();
    await model.loadModel();
    const articles = loadArticles();
    const realTrend = 2;

    let consecutiveSuccesses = 0;
    let episode = 1;
    const maxEpisodes = 20;
    let finalResults = {
        success: false,
        realTrend: realTrend,
        totalEpisodes: 0,
        decisionHistory: []
    };

    while (consecutiveSuccesses < 3 && episode <= maxEpisodes) {
        console.log("-------------------------------------------------");
        console.log(`Début de l'épisode`);
        const decision = await model.checkPrediction(articles, realTrend);
        console.log("Épisode : ", episode++);
        consecutiveSuccesses = (decision === realTrend) ? consecutiveSuccesses + 1 : 0;

        console.log(`Succès consécutifs : ${consecutiveSuccesses}`);

        finalResults.decisionHistory.push({ episode, decision });

        if (consecutiveSuccesses >= 3 && model.experienceReplayMemory.length < 5) {
            console.log("Le modèle a réussi à prédire correctement 3 fois consécutivement.");
            finalResults.success = true;
            break;
        }
        console.log("-------------------------------------------------");
    }

    if (consecutiveSuccesses < 3) {
        console.log("Le modèle n'a pas réussi à faire 3 prédictions correctes consécutives.");
        finalResults.success = false;
    }

    if (episode > maxEpisodes) {
        console.log("La limite d'épisodes a été atteinte.");
        finalResults.success = false;
    }

    finalResults.totalEpisodes = episode - 1;  

    return finalResults;
};
