import os
import sys
import time
import argparse
import logging
import urllib.request
import urllib.parse
import json
import re
import xml.etree.ElementTree as ET
from dotenv import load_dotenv

# Set up logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Load environment variables
load_dotenv()

# Track already posted articles to avoid repetition
HISTORY_FILE = "posted_articles.txt"

def load_posted_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def save_to_posted_history(link):
    with open(HISTORY_FILE, "a") as f:
        f.write(link + "\n")

def fetch_latest_ai_news():
    """
    Fetches the top AI news from Google News RSS feed.
    """
    rss_url = "https://news.google.com/rss/search?q=Artificial+Intelligence&hl=en-US&gl=US&ceid=US:en"
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(rss_url, headers=headers)
    
    logging.info("Fetching latest AI news from RSS feed...")
    try:
        with urllib.request.urlopen(req) as response:
            xml_data = response.read()
            
        root = ET.fromstring(xml_data)
        items = []
        for item in root.findall(".//item"):
            title = item.find("title").text
            link = item.find("link").text
            pub_date = item.find("pubDate").text
            items.append({
                "title": title,
                "link": link,
                "pub_date": pub_date
            })
        logging.info(f"Successfully fetched {len(items)} items from RSS.")
        return items
    except Exception as e:
        logging.error(f"Error fetching AI news RSS: {e}")
        return []

def query_openrouter(prompt, model="deepseek/deepseek-chat"):
    """
    Queries OpenRouter API for content and image prompt generation.
    """
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"}
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            res_body = json.loads(response.read().decode("utf-8"))
            return res_body["choices"][0]["message"]["content"]
    except Exception as e:
        logging.error(f"OpenRouter connection error: {e}")
        return None

def query_ollama(prompt, model="deepseek-r1:14b"):
    """
    Queries local Ollama instance for content and image prompt generation.
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            res_body = json.loads(response.read().decode("utf-8"))
            return res_body.get("response", "")
    except Exception as e:
        logging.error(f"Ollama connection error (is Ollama running?): {e}")
        return None

def clean_thinking_tags(text):
    """
    Removes <think>...</think> reasoning blocks if returned by DeepSeek R1 models.
    """
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned.strip()

def extract_json(text):
    """
    Extracts raw JSON content from markdown code blocks if present.
    """
    text = text.strip()
    # Check if wrapped in ```json and ```
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Check if wrapped in plain ``` and ```
    match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text

def synthesize_marketing_content(news_item, model="deepseek-r1:14b"):
    """
    Formulates the tweet caption and a Stable Diffusion visual prompt.
    """
    prompt = f"""
    Analyze this news update:
    Title: {news_item['title']}
    Link: {news_item['link']}

    You are a marketing strategist for a trainer specializing in "Agentic Engineering" and "AI Agents".
    Compose a response in JSON format containing two fields:
    1. "tweet": A compelling, informative tweet summarizing this news, explaining why it matters for agentic AI/LLM development, and ending with a creative call-to-action promoting your "Agentic Engineering Training". The entire tweet MUST be under 260 characters (including spaces and hashtags).
    2. "image_prompt": A detailed prompt for a text-to-image generator (Stable Diffusion) to create a matching high-quality visual representation of the news. Focus on futuristic, sleek, professional tech designs (e.g. 3D renders, neon wireframes, digital dashboards). Do not use text in the image.

    Format the final output strictly as JSON with keys "tweet" and "image_prompt".
    """
    
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "")
    is_openrouter = api_key and ("sk-or-" in api_key or "openrouter" in base_url)
    
    if is_openrouter:
        or_model = "deepseek/deepseek-chat"
        logging.info(f"Synthesizing marketing post using OpenRouter model '{or_model}'...")
        raw_response = query_openrouter(prompt, or_model)
    else:
        logging.info(f"Synthesizing marketing post using local Ollama model '{model}'...")
        raw_response = query_ollama(prompt, model)
        
    if not raw_response:
        return None
        
    cleaned_response = clean_thinking_tags(raw_response)
    json_content = extract_json(cleaned_response)
    
    try:
        data = json.loads(json_content)
        return data
    except Exception as e:
        logging.error(f"Failed to parse JSON response: {e}")
        logging.info(f"Raw response was:\n---\n{json_content}\n---")
        return None

def generate_image_locally(image_prompt, output_path="temp_post_image.png"):
    """
    Generates an image using Stable Diffusion locally on Apple Silicon (MPS).
    """
    logging.info("Initializing local Stable Diffusion image generator on GPU (MPS)...")
    try:
        import torch
        from diffusers import AutoPipelineForText2Image
        
        # We use sd-turbo because it is blazingly fast and requires very little VRAM/steps
        model_id = "stabilityai/sd-turbo"
        logging.info(f"Loading pipeline for '{model_id}'...")
        
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            variant="fp16",
            use_safetensors=True,
            local_files_only=True
        )
        
        # Set device to Metal Performance Shaders (Apple Silicon GPU)
        pipe.to("mps")
        
        logging.info(f"Generating image with prompt: '{image_prompt}'...")
        # sd-turbo works perfectly in 1 or 2 steps
        start_time = time.time()
        result = pipe(
            prompt=image_prompt, 
            num_inference_steps=2, 
            guidance_scale=0.0
        )
        image = result.images[0]
        image.save(output_path)
        
        duration = time.time() - start_time
        logging.info(f"Successfully generated and saved local image to '{output_path}' in {duration:.1f} seconds.")
        return True
    except Exception as e:
        logging.error(f"Error generating local image: {e}")
        return False

def post_to_x(tweet_text, image_path, dry_run=False):
    """
    Uploads the image and posts the tweet to your X (Twitter) account.
    """
    if dry_run:
        logging.info(f"[DRY-RUN] Would have uploaded image '{image_path}' to X.")
        logging.info(f"[DRY-RUN] Would have posted tweet:\n---\n{tweet_text}\n---")
        return True
        
    logging.info("Initializing connection to X API...")
    try:
        import tweepy
        
        consumer_key = os.getenv("X_CONSUMER_KEY")
        consumer_secret = os.getenv("X_CONSUMER_SECRET")
        access_token = os.getenv("X_ACCESS_TOKEN")
        access_token_secret = os.getenv("X_ACCESS_TOKEN_SECRET")
        
        if not all([consumer_key, consumer_secret, access_token, access_token_secret]):
            logging.error("Missing X API keys in environment or .env file.")
            return False
            
        # Authentication for Media Upload (V1.1 API)
        auth = tweepy.OAuth1UserHandler(
            consumer_key, 
            consumer_secret, 
            access_token, 
            access_token_secret
        )
        api = tweepy.API(auth)
        
        # Authentication for Tweet Posting (V2 API)
        client = tweepy.Client(
            consumer_key=consumer_key,
            consumer_secret=consumer_secret,
            access_token=access_token,
            access_token_secret=access_token_secret
        )
        
        logging.info(f"Uploading media '{image_path}' to X...")
        media = api.media_upload(image_path)
        logging.info(f"Media uploaded successfully. Media ID: {media.media_id}")
        
        logging.info("Posting tweet...")
        response = client.create_tweet(
            text=tweet_text, 
            media_ids=[media.media_id]
        )
        logging.info("Tweet posted successfully!")
        return True
    except Exception as e:
        logging.error(f"Error posting to X: {e}")
        return False

def post_to_discord(text, image_path, dry_run=False):
    """
    Optional: Posts the message and image to a Discord channel via incoming Webhook.
    """
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        return True
        
    if dry_run:
        logging.info(f"[DRY-RUN] Would have posted to Discord webhook.")
        return True
        
    logging.info("Publishing to Discord webhook...")
    try:
        import requests
        with open(image_path, "rb") as f:
            files = {"file": f}
            data = {"content": text}
            response = requests.post(webhook_url, files=files, data=data)
            if response.status_code in [200, 204]:
                logging.info("Successfully posted to Discord!")
                return True
            else:
                logging.error(f"Failed to post to Discord: {response.status_code} - {response.text}")
                return False
    except Exception as e:
        logging.error(f"Error posting to Discord: {e}")
        return False

def post_to_telegram(text, image_path, dry_run=False):
    """
    Optional: Posts the message and image to a Telegram channel/group via Bot API.
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        return True
        
    if dry_run:
        logging.info(f"[DRY-RUN] Would have posted to Telegram chat {chat_id}.")
        return True
        
    logging.info(f"Publishing to Telegram chat {chat_id}...")
    try:
        import requests
        url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
        with open(image_path, "rb") as f:
            files = {"photo": f}
            data = {
                "chat_id": chat_id,
                "caption": text
            }
            response = requests.post(url, files=files, data=data)
            if response.status_code == 200:
                logging.info("Successfully posted to Telegram!")
                return True
            else:
                logging.error(f"Failed to post to Telegram: {response.status_code} - {response.text}")
                return False
    except Exception as e:
        logging.error(f"Error posting to Telegram: {e}")
        return False

def execute_cycle(model="deepseek-r1:14b", dry_run=False):
    """
    Main sequence of a single marketing cycle.
    """
    posted_history = load_posted_history()
    news_items = fetch_latest_ai_news()
    
    if not news_items:
        logging.warning("No news items retrieved. Skipping cycle.")
        return False
        
    # Find the latest news article that hasn't been posted yet
    target_news = None
    for item in news_items:
        if item["link"] not in posted_history:
            target_news = item
            break
            
    if not target_news:
        logging.info("All fetched news articles have already been posted. Trying the absolute latest update to refresh.")
        target_news = news_items[0]
        
    logging.info(f"Selected article for processing: '{target_news['title']}'")
    
    # Step 1: Synthesize marketing caption and image prompt using Ollama
    synthesized = synthesize_marketing_content(target_news, model=model)
    if not synthesized:
        logging.error("Content synthesis failed. Skipping cycle.")
        return False
        
    tweet_text = f"{synthesized['tweet']}\n\nRead more: {target_news['link']}"
    image_prompt = synthesized['image_prompt']
    
    # Step 2: Generate visual assets locally on GPU (MPS)
    image_path = "dry_run_post.png" if dry_run else "marketing_post.png"
    img_success = generate_image_locally(image_prompt, image_path)
    if not img_success:
        logging.error("Local image generation failed. Skipping cycle.")
        return False
        
    # Step 3: Publish to X (Twitter)
    post_success = post_to_x(tweet_text, image_path, dry_run=dry_run)
    
    # Optional Step 4: Publish to Discord and Telegram
    post_to_discord(tweet_text, image_path, dry_run=dry_run)
    post_to_telegram(tweet_text, image_path, dry_run=dry_run)
    
    if post_success and not dry_run:
        save_to_posted_history(target_news["link"])
        logging.info("Cycle completed successfully and logged to history.")
        return True
    elif dry_run:
        logging.info("Dry-run cycle completed successfully. Local files generated.")
        return True
    else:
        logging.error("Failed to post update. Cycle failed.")
        return False

def main():
    parser = argparse.ArgumentParser(description="24/7 Local X Marketing Agent Daemon")
    parser.add_argument("--dry-run", action="store_true", help="Run a single local cycle without posting to X")
    parser.add_argument("--model", type=str, default="deepseek-r1:14b", help="Ollama LLM model name")
    parser.add_argument("--interval", type=int, default=3600, help="Interval in seconds between cycles (default: 3600 / 1 hour)")
    args = parser.parse_args()
    
    logging.info("==============================================")
    logging.info("Starting 24/7 Local X Marketing Agent Daemon")
    logging.info(f"Ollama model: {args.model}")
    logging.info(f"Mode: {'Dry-Run (Single Execution)' if args.dry_run else f'Daemon Mode (Interval: {args.interval}s)'}")
    logging.info("==============================================")
    
    if args.dry_run:
        execute_cycle(model=args.model, dry_run=True)
        logging.info("Dry run finished.")
        sys.exit(0)
        
    # Daemon loop
    while True:
        try:
            execute_cycle(model=args.model, dry_run=False)
        except Exception as e:
            logging.error(f"Error in daemon cycle: {e}")
            
        logging.info(f"Waiting for {args.interval} seconds before next cycle...")
        time.sleep(args.interval)

if __name__ == "__main__":
    main()
