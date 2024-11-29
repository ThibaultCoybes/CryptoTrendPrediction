import fetch from 'node-fetch'; 
import fs from 'fs'; 

const apiKey = '2bb8d107c56749208876a29c39e58857';

const params = {
    q: 'dogecoin',  
    from: '2024-11-07', 
    to: '2024-111-10', 
    apiKey: apiKey,
    pageSize: 50, 
    language: 'en' 
};

const url = new URL('https://newsapi.org/v2/everything');
url.search = new URLSearchParams(params).toString();

async function fetchArticles() {
    try {
        const response = await fetch(url);
        const data = await response.json();
        
        if (data.status === 'ok') {
            let articles = data.articles.map(article => ({
                content: article.content ? article.content.slice(0, 400) : '' 
            }));

            articles = articles.map(article => ({
                ...article,
                content: article.content.replace(/<\/?[^>]+(>|$)/g, "") 
            }));

            const fileName = 'dogecoin_articles_gain.json';
            fs.writeFileSync(fileName, JSON.stringify(articles, null, 4), 'utf8');
            console.log(`Fichier ${fileName} créé avec succès.`);
        } else {
            console.log('Erreur dans la récupération des articles:', data.message);
        }
    } catch (error) {
        console.error('Erreur de fetch:', error);
    }
}

fetchArticles();
