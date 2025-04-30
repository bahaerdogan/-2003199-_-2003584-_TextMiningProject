import requests
from bs4 import BeautifulSoup
import time
import random
import csv

headers = {
    "User-Agent": "Mozilla/5.0"
}

films = {
    "Blue Ruin": "https://letterboxd.com/film/blue-ruin/",
    "The Rover": "https://letterboxd.com/film/the-rover-2014/",
    "Cold in July": "https://letterboxd.com/film/cold-in-july/",
    "A Hijacking": "https://letterboxd.com/film/a-hijacking/",
    "Loveless": "https://letterboxd.com/film/loveless-2017/",
    "Victoria": "https://letterboxd.com/film/victoria-2015/",
    "Timecrimes": "https://letterboxd.com/film/timecrimes/",
    "The Guilty": "https://letterboxd.com/film/the-guilty-2018/",
    "Calibre": "https://letterboxd.com/film/calibre/",
    "Coherence": "https://letterboxd.com/film/coherence/",
    "Good Time": "https://letterboxd.com/film/good-time/",
    "The Invitation": "https://letterboxd.com/film/the-invitation-2015/",
    "Ida": "https://letterboxd.com/film/ida/",
    "The Painted Bird": "https://letterboxd.com/film/the-painted-bird/",
    "Leave No Trace": "https://letterboxd.com/film/leave-no-trace/",
    "Children of Men": "https://letterboxd.com/film/children-of-men/",
    "The Handmaiden": "https://letterboxd.com/film/the-handmaiden/",
    "Oldboy": "https://letterboxd.com/film/oldboy/",
    "The Master": "https://letterboxd.com/film/the-master-2012/",
    "Pan’s Labyrinth": "https://letterboxd.com/film/pans-labyrinth/",
    "Heat": "https://letterboxd.com/film/heat/",
    "Memories of Murder": "https://letterboxd.com/film/memories-of-murder/",
    "The Raid": "https://letterboxd.com/film/the-raid/",
    "There Will Be Blood": "https://letterboxd.com/film/there-will-be-blood/",
    "The Assassin": "https://letterboxd.com/film/the-assassin-2015/"
    
}


output_data = []

def scrape_reviews(film_name, base_url, sort_type, target_count=15):
    comments = []
    page = 1

    while len(comments) < target_count * 2:
        if sort_type == "lowest":
            url = base_url.rstrip("/") + f"/reviews/by/entry-rating-lowest/page/{page}/"
        elif sort_type == "highest":
            url = base_url.rstrip("/") + f"/reviews/by/entry-rating/page/{page}/"
        else:
            raise ValueError("Sort type must be 'lowest' or 'highest'.")

        print(f"Scraping {film_name} - {sort_type} - Page {page}...")

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
        output_data.append({
            "film": film_name,
            "sort_type": sort_type,
            "review": comment
        })

for film_name, film_url in films.items():
    scrape_reviews(film_name, film_url, sort_type="lowest", target_count=15)
    scrape_reviews(film_name, film_url, sort_type="highest", target_count=15)

with open("AllCommentsBalanced.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["film", "sort_type", "review"])
    writer.writeheader()
    writer.writerows(output_data)

print("Scraping completed and saved to AllCommentsBalanced.csv!")
