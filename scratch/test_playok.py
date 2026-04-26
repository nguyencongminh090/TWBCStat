import requests
from bs4 import BeautifulSoup

def fetch_playok_stats(url):
    print(f"Fetching {url}...")
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')
    
    wins = 0
    draws = 0
    losses = 0
    
    for tr in soup.find_all('tr'):
        cells = [td.text.strip() for td in tr.find_all('td')]
        if len(cells) >= 3:
            res = cells[2].lower()
            if res == 'win': wins += 1
            elif res == 'loss': losses += 1
            elif res == 'draw': draws += 1
            
    print(f"Stats: Wins: {wins}, Draws: {draws}, Losses: {losses}")
    return wins, draws, losses

fetch_playok_stats('https://www.playok.com/en/stat.phtml?u=wbcbb&g=gm&sk=2&oid=wbcdeafbat')
