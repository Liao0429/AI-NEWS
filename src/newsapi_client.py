import os
import requests
from dotenv import load_dotenv


def get_newsapi_api_key():
    """从环境变量获取NewsAPI API key"""
    load_dotenv()
    api_key = os.getenv('NEWSAPI_API_KEY')
    if not api_key:
        raise ValueError('NEWSAPI_API_KEY environment variable is not set')
    return api_key


def fetch_news_from_newsapi(query, from_date, to_date, page_size=100):
    """使用环境变量中的API key从NewsAPI抓取新闻"""
    api_key = get_newsapi_api_key()
    url = 'https://newsapi.org/v2/everything'
    
    params = {
        'q': query,
        'from': from_date,
        'to': to_date,
        'sortBy': 'publishedAt',
        'pageSize': page_size,
        'apiKey': api_key,
        'language': 'en'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get('articles', [])
    except requests.exceptions.RequestException as e:
        print(f'Error fetching news: {e}')
        return []
