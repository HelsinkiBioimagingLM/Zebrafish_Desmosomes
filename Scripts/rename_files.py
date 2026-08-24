import os
import sys
from pathlib import Path
from tqdm import tqdm

files = list(Path(sys.argv[1]).rglob('*.czi'))
print('replacing spaces in file paths...')
for f in tqdm(files):
    new_name = Path(str(f).replace(' ','_'))
    #print(f'{new_name}')
    os.rename(f,new_name)
    #break
print('Done!')