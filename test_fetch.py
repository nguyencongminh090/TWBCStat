import requests
import re
from crawler import strip_flags

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

url = "https://sites.google.com/view/worldblitzcup/twbc-2026/matches/round-1/hungary-b-vs-hungary-a"
r = requests.get(url, headers=HEADERS)
text = re.sub(r'<[^>]+>', ' ', r.text)
text = re.sub(r'\s+', ' ', text)
text = strip_flags(text)
print("TEXT PREFIX:", text[:500])
print("TEXT SUFFIX:", text[-500:])

title_iters = list(re.finditer(r'Team World Blitz Championship', text))
if title_iters:
    print("FOUND TITLE!")
else:
    print("TITLE NOT FOUND!")
    
match_headers = list(re.finditer(r'ROUND', text))
print("ROUNDS FOUND:", len(match_headers))
