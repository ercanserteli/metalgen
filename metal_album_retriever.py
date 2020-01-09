import time
import requests
import json

    
def get_album_ids():
    album_query = "https://musicbrainz.org/ws/2/release-group?query=tag:metal&limit=100&offset={}&fmt=json"

    album_ids = []

    for i in range(100):
        r = requests.get(album_query.format(i*100))
        albums = r.json()["release-groups"]
        album_ids.extend([a["id"] for a in albums])
        time.sleep(1)

    with open("album_ids.json", "w") as f:
        json.dump(album_ids, f)
        
        
def get_album_covers():
    cover_query = "https://coverartarchive.org/release-group/{}/front-250"
    last_i_filename = "covers/last_i.txt"
    
    with open("album_ids.json", "r") as f:
        album_ids = json.load(f)
    with open("covers/errors.json", "r") as f:
        error_ids = json.load(f)
    
    with open(last_i_filename, "r") as f:
        last_i = int(f.read())
    
    
    for i, id in enumerate(album_ids[last_i+1:]):
        try:
            r = requests.get(cover_query.format(id))
        except:
            error_ids.append(i)
            with open("covers/errors.json", "w") as f:
                json.dump(error_ids, f)
        else:
            if r.status_code == 200:
                open('covers/{}.jpg'.format(id), 'wb').write(r.content)
            else:
                print("Got code: ", r.status_code)
                
            with open(last_i_filename, "w") as f:
                f.write(str(i+last_i+1))
        time.sleep(0.1)


def get_album_titles():
    album_query = "https://musicbrainz.org/ws/2/release-group?query=tag:metal&limit=100&offset={}&fmt=json"

    album_ids = []

    for i in range(100):
        r = requests.get(album_query.format(i*100))
        albums = r.json()["release-groups"]
        album_ids.extend([{a["id"]: a["title"]} for a in albums])
        time.sleep(1)

    with open("album_id_titles.json", "w") as f:
        json.dump(album_ids, f)

if __name__ == "__main__":
    get_album_titles()