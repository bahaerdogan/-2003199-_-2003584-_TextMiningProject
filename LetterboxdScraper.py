# LetterboxdScraper.py
import requests
from bs4 import BeautifulSoup
import time
import random
import csv

headers = {
    "User-Agent": "Mozilla/5.0"
}

films = {
    "Interstellar": "https://letterboxd.com/film/interstellar",
    "The Dark Knight": "https://letterboxd.com/film/the-dark-knight",
    "Inception": "https://letterboxd.com/film/inception",
    "Lord of War": "https://letterboxd.com/film/lord-of-war",
    "The Zone of Interest": "https://letterboxd.com/film/the-zone-of-interest",
    "Moneyball": "https://letterboxd.com/film/moneyball",
    "The Last of the Mohicans": "https://letterboxd.com/film/the-last-of-the-mohicans",
    "The Butterfly Effect": "https://letterboxd.com/film/the-butterfly-effect",
    "Sicario": "https://letterboxd.com/film/sicario",
    "The Hangover": "https://letterboxd.com/film/the-hangover",
    "Mission: Impossible - Fallout": "https://letterboxd.com/film/mission-impossible-fallout",
    "The Game": "https://letterboxd.com/film/the-game",
    "Toy Story 4": "https://letterboxd.com/film/toy-story-4",
    "The Last King of Scotland": "https://letterboxd.com/film/the-last-king-of-scotland",
    "Enemy at the Gates": "https://letterboxd.com/film/enemy-at-the-gates",
    "Fight Club": "https://letterboxd.com/film/fight-club",
    "John Wick": "https://letterboxd.com/film/john-wick",
    "The Matrix": "https://letterboxd.com/film/the-matrix",
    "11 Rebels": "https://letterboxd.com/film/11-rebels",
    "Top Gun": "https://letterboxd.com/film/top-gun",
}

output_data = []

def scrape_reviews(film_name, film_url, target_count=25):
    comments = []
    page = 1

    while len(comments) < target_count * 2:
        url = film_url.rstrip("/") + f"/reviews/page/{page}/"
        print(f"Scraping {film_name} - Page {page}...")
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to fetch {url}")
            break
        
        soup = BeautifulSoup(response.text, "html.parser")
        review_blocks = soup.find_all("div", class_="body-text")

        if not review_blocks:
            print(f"No more reviews found on page {page} for {film_name}.")
            break

        for review in review_blocks:
            text = review.get_text(strip=True)
            if text and len(text) > 30:
                comments.append(text)
        
        page += 1
        time.sleep(random.uniform(0.5, 1.5))

    random.shuffle(comments)
    selected_comments = comments[:target_count]

    for comment in selected_comments:
        output_data.append({"film": film_name, "review": comment})

for film_name, film_url in films.items():
    scrape_reviews(film_name, film_url)

# Tüm yorumları kaydet (filtrelenmemiş)
with open("AllComments.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["film", "review"])
    writer.writeheader()
    writer.writerows(output_data)

print("Scraping completed and saved to AllComments.csv!")
