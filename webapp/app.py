from flask import Flask, render_template, request
from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
from google_play_scraper import reviews
from app_store_scraper import AppStore
import re

app = Flask(__name__)  

################################################################################################
# Sikayetvar
################################################################################################

headers = {'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.28 Safari/537.36'}

def sikayetvar_get_complaint_links(url):
    yorum_linkleri = []
    page_url = f"{url}?page=1"
    req = Request(page_url, headers=headers)
    page_request = urlopen(req)
    soup = BeautifulSoup(page_request.read(), "html.parser")
    containers = soup.find_all("article", class_="card-v2 card-v3 ga-v ga-c")
    for i in range(3):
        a_div = containers[i].find("a", class_="complaint-layer card-v3-container")
        data_url = a_div['href']
        print(data_url)
        yorum_linkleri.append("https://www.sikayetvar.com" + data_url)
    return yorum_linkleri
    
def sikayetvar_scrape_complaint(url):
    req = Request(url, headers=headers)
    page_request = urlopen(req)
    soup = BeautifulSoup(page_request, "html.parser")
    try:
        yorum = soup.find("div", class_="complaint-detail-description").text.strip()
        return yorum
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None

def sikayetvar_find(search):
    url = f"https://www.sikayetvar.com/{search}"
    review_links = sikayetvar_get_complaint_links(url)
    reviews_arr = []

    for idx, link in enumerate(review_links, 1):
        complaint_data = sikayetvar_scrape_complaint(link)
        reviews_arr.append({f"{idx}": complaint_data})

    return reviews_arr

################################################################################################

################################################################################################
# Google Play Store
################################################################################################

def playstore_get_reviews(url):
    result = reviews(url, lang="tr", country="tr", count=3)
    reviews_arr = []
    for idx, r in enumerate(result[0]):
        reviews_arr.append({f"{idx}": r["content"]})

    return reviews_arr


def playstore_find(search):
    search_json = {
        "paycell": "com.turkcell.paycell",
        "turkcell": "com.ttech.android.onlineislem",
        "bip": "com.turkcell.bip",
        "gnç": "com.solidict.gnc2",
        "upcall": "com.turkcell.sesplus",
        "platinum": "com.turkcellplatinum.main",
        "calarkendinlet": "tr.com.turkcell.calarkendinlet",
        "tv+": "com.turkcell.ott",
        "fizy": "com.turkcell.gncplay",
        "lifebox": "tr.com.turkcell.akillidepo"
    }

    url = search_json[search]
    result = playstore_get_reviews(url)
    return result

################################################################################################

################################################################################################
# App Store
################################################################################################

def appstore_get_reviews(app_name, app_id):
    data = AppStore(country='tr', app_name=app_name, app_id=app_id)
    data.review(how_many=1)

    reviews_arr = []
    print(data.reviews)
    print(len(data.reviews))
    for idx, r in enumerate(data.reviews[:3]):
        reviews_arr.append({f"{idx}": r["review"]})

    return reviews_arr

def appstore_find(search):
    search_json = {
        "paycell": { "app_name": "Paycell - Digital Wallet", "app_id": 1198609962 },
        "turkcell": { "app_name": "Turkcell", "app_id": 335162906 },
        "bip": { "app_name": "BiP - Messenger, Video Call", "app_id": 583274826 },
        "gnç": { "app_name": "GNÇ", "app_id": 894318685 },
        "upcall": { "app_name": "UpCall", "app_id": 1149307476 },
        "platinum": { "app_name": "Turkcell Platinum", "app_id": 671494224 },
        "calarkendinlet": { "app_name": "ÇalarkenDinlet", "app_id": 1026830839 },
        "tv+": { "app_name": "TV+", "app_id": 835880015 },
        "fizy": { "app_name": "fizy – Music & Video", "app_id": 404239912 },
        "lifebox": { "app_name": "Lifebox: Storage & Backup", "app_id": 665036334 }
    }

    app_name = search_json[search]["app_name"]
    app_id = search_json[search]["app_id"]
    result = appstore_get_reviews(app_name, app_id)
    return result

################################################################################################


@app.route("/")
def main():
    return render_template('./home.html')

@app.route("/predict", methods=['POST'])
def home():
    search = request.form['search']
    firm = request.form["firm"]
    if firm == "sikayetvar":
        reviews_arr = sikayetvar_find(search)
    elif firm == "playstore":
        reviews_arr = playstore_find(search)
    elif firm == "appstore":
        reviews_arr = appstore_find(search)

    # Başlığı dinamik olarak ayarla
    title = f"{firm} sitesinde {search} firmasi için yapılan yorumlar"
    print(reviews_arr)
    def clean_comment(comment_dict):
        comment = next(iter(comment_dict.values()))
        return re.sub(r"\{'\d+':\s'(.+)'\}", r"\1", comment)
    reviews_arr = [clean_comment(review) for review in reviews_arr]
    return render_template('./after.html', data=reviews_arr, title=title, search=search)

app.run(port=5000, debug=False)
