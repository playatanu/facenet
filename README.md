# FaceNet - PyTorch

The code is tested using torch 2.3.0 under Windows11 with Python 3.9.0

#

- Install:
```bash
git clone https://github.com/playatanu/facenet.git facenet
```

```bash
pip install -r requirements.txt
```

- FaceNet Example:
```python
from facenet import FaceNet

# Load model
model = FaceNet('model.pt')

# Train the model
model.train('path/to/train')

# Predict 
result = model('path/to/image.jpg')
```





