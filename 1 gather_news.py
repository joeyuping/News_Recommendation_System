def gather_news_data(keyword, api_key, from_date=None, to_date=None, language='en', sort_by='relevancy', page_size=100):
    """
    Gather news data about a specific keyword using newsapi
    
    Args:
        keyword (str): Keyword to search for (e.g., 'earthquake', 'hurricane')
        api_key (str): NewsAPI API key
        from_date (str, optional): Start date in format 'YYYY-MM-DD'
        to_date (str, optional): End date in format 'YYYY-MM-DD'
        language (str, optional): Language of articles
        sort_by (str, optional): Sort articles by 'relevancy', 'popularity', or 'publishedAt'
        page_size (int, optional): Number of results per page (max 100)
        
    Returns:
        list: List of news articles
    """
    import requests
    import json
    from datetime import datetime, timedelta
    import os
    
    # Set default dates if not provided
    if not from_date:
        from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    if not to_date:
        to_date = datetime.now().strftime('%Y-%m-%d')
    
    # Construct URL
    url = f"https://newsapi.org/v2/everything?q={keyword}&from={from_date}&to={to_date}&language={language}&sortBy={sort_by}&pageSize={page_size}&apiKey={api_key}"
    
    # Make request
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        articles = data.get('articles', [])
        
        # Create a filename with the keyword and date
        filename = f"news_{keyword}_{datetime.now().strftime('%Y%m%d')}.json"
        
        # Save the data to a JSON file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=4)
            
        print(f"Saved {len(articles)} articles to {filename}")
        return articles
    else:
        print(f"Error: {response.status_code}")
        return []

# Call the function with your parameters
if __name__ == "__main__":
    gather_news_data('地震災情與救援行動', 'f4b6348f8e5740b7a0a0fde0b65d2573', 
                     from_date='2025-03-14', to_date='2025-04-13', 
                     language='zh', sort_by='relevancy', page_size=100)
