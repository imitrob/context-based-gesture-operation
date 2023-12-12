
from pathlib import Path
import gdown
# import tarfile

if __name__ == '__main__':
    path = Path(__file__)
    
    # 1    
    url_model = "https://drive.google.com/uc?id=1miZZH9db0C7EG06j72JrZJe7Gkd6qAVV"
    out_model = path.parent.joinpath('trained_networks', 'M3v10_D6.pkl')
    path.parent.joinpath('trained_networks').mkdir(exist_ok=True)
    
    input(f"Download sample network to {out_model}")
    gdown.download(url_model, str(out_model), quiet=False)