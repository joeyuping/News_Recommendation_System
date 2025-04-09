import json
import requests
from bs4 import BeautifulSoup
import time
import os

def scrape_article(url):
    """
    Scrape the full content of an article from its URL
    
    Args:
        url (str): URL of the article to scrape
        
    Returns:
        dict: Dictionary containing the scraped article data
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract article title
        title = soup.find('h1')
        title_text = title.text.strip() if title else "Title not found"
        
        # Try different selectors for article content
        # This is a generic approach - specific websites might need custom selectors
        content_selectors = [
            'article', 
            '.article-content',
            '.article-body',
            '.story-body',
            '.content-body',
            '#article-body',
            '.post-content',
            '.entry-content',
            '.story',
            '.content'
        ]
        
        content = None
        for selector in content_selectors:
            content_element = soup.select_one(selector)
            if content_element and len(content_element.text.strip()) > 100:
                content = content_element
                break
        
        if not content:
            # If no content found with selectors, try paragraphs within the main tag
            main = soup.find('main')
            if main:
                paragraphs = main.find_all('p')
                content_text = ' '.join([p.text for p in paragraphs])
            else:
                # Last resort: get all paragraphs
                paragraphs = soup.find_all('p')
                content_text = ' '.join([p.text for p in paragraphs if len(p.text.strip()) > 40])
        else:
            # Clean up the content
            # Remove scripts, styles, and other non-content elements
            for element in content.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()
            
            # Get all paragraphs from the content
            paragraphs = content.find_all('p')
            if paragraphs:
                content_text = ' '.join([p.text for p in paragraphs])
            else:
                content_text = content.text
        
        # Clean up the content text
        content_text = content_text.strip()
        
        return {
            'title': title_text,
            'url': url,
            'full_content': content_text,
            'scrape_success': True
        }
    
    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return {
            'url': url,
            'scrape_success': False,
            'error': str(e)
        }

def process_news_json(input_file, output_file=None):
    """
    Process a JSON file of news articles from NewsAPI and scrape the full content
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str, optional): Path to the output JSON file. If None, will use input_file with '_full' suffix
        
    Returns:
        list: List of articles with full content
    """
    # Set default output file if not provided
    if not output_file:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_full{ext}"
    
    # Load the news articles from the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    # Process each article
    full_articles = []
    for i, article in enumerate(articles):
        print(f"Processing article {i+1}/{len(articles)}: {article.get('title', 'No title')}")
        
        # Get the URL
        url = article.get('url')
        if not url:
            print(f"No URL found for article {i+1}")
            article['full_content'] = None
            article['scrape_success'] = False
            full_articles.append(article)
            continue
        
        # Scrape the article
        scraped_data = scrape_article(url)
        
        # Add the scraped data to the article
        article['full_content'] = scraped_data.get('full_content')
        article['scrape_success'] = scraped_data.get('scrape_success', False)
        
        # Add the article to the list
        full_articles.append(article)
        
        # Sleep to avoid overloading the server
        time.sleep(1)
    
    # Save the full articles to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(full_articles, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(full_articles)} articles with full content to {output_file}")
    return full_articles

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Scrape full content of news articles from a NewsAPI JSON file')
    parser.add_argument('input_file', help='Path to the input JSON file')
    parser.add_argument('--output-file', '-o', help='Path to the output JSON file')
    
    args = parser.parse_args()
    
    process_news_json(args.input_file, args.output_file)
