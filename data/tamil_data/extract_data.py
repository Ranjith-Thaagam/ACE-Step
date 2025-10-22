import os
import pandas as pd 
from collections import defaultdict

try:
    loc = os.path.join(os.getcwd(),"data/tamil_data")
    path=os.path.join(loc,"songs")

    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")

    files_path = os.listdir(path)
   

    audio_files = [f for f in files_path if f.endswith((".wav",".mp3"))]
    lyric_files = [f for f in files_path if f.endswith('_lyric.txt')]
    prompt_files = [f for f in files_path if f.endswith('_prompt.txt')]

    songs_dict = defaultdict(dict)
    for f in files_path:
        name, ext = os.path.splitext(f)
        if ext in ['.wav', '.mp3']:
            songs_dict[name]['audio'] = f
        elif '_lyric' in name:
            base = name.replace('_lyric', '')
            songs_dict[base]['lyric'] = f
        elif '_prompt' in name:
            base = name.replace('_prompt', '')
            songs_dict[base]['prompt'] = f

    dataset={
        "song_name":[],
        "audio_path":[],
        "lyrics_path":[],
        "prompt_path":[],
    }
    for song, files in list(songs_dict.items()):  # first 5 songs
        dataset['song_name'].append(song)
        dataset['audio_path'].append(files.get('audio',None))
        dataset['lyrics_path'].append(files.get('lyric',None))
        dataset['prompt_path'].append(files.get('prompt',None))


    data = pd.DataFrame(dataset)
    data.to_csv(f'{loc}\\metadata.csv',index=False)
    print(data)
    print("Success")

except Exception as e:
    print("Error : ",e)
