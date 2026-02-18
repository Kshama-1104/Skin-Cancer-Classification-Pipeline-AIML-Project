## Deploy local PyTorch model with a simple HTML UI

### 1) Setup

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

If your model file isn't named `model.pt`, set `MODEL_PATH` when starting the server.

### 2) Run

```powershell
# If your file is TorchScript already
python server.py

# Or specify env for size/normalization or scripted file
$env:MODEL_PATH='model_scripted.pt'
$env:IMAGE_SIZE='224'
$env:MEAN='0.485,0.456,0.406'
$env:STD='0.229,0.224,0.225'
python server.py
```

Open `http://localhost:8000` and upload an image.

### Notes

- The server loads the Torch model from `model.pt` on CPU and sets `eval()`.
- Preprocessing resizes images to 224x224 and performs a simple normalization to [0,1]. Adjust `preprocess_image` and `postprocess_output` in `server.py` to match your training pipeline.

#### If you see "OrderedDict object is not callable"

This means the file is a `state_dict` (only weights) and not a runnable model. For deployment either:

1) Rebuild your model in code and load weights:

```python
model = MyNet(...)
model.load_state_dict(torch.load('model.pt', map_location='cpu'))
model.eval()
torch.jit.script(model).save('model_scripted.pt')
```

2) Or export directly after training to TorchScript and deploy that `.pt`:

```python
scripted = torch.jit.script(trained_model.eval())
scripted.save('model_scripted.pt')
```

Then set `MODEL_PATH=model_scripted.pt` when starting the server.

### Export your checkpoint to TorchScript

1) Install dependencies (adds NumPy too):

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

2) Open `export_to_torchscript.py` and implement `build_model(num_classes)` to match your training architecture.

3) Run the exporter (replace num-classes with your value). You can use presets or your custom build:

```powershell
# Using a preset backbone (no code edits):
python export_to_torchscript.py --checkpoint model.pt --out model_scripted.pt --image-size 224 --num-classes 1000 --preset resnet18

# Or using your own build_model implementation:
python export_to_torchscript.py --checkpoint model.pt --out model_scripted.pt --image-size 224 --num-classes 1000
```

ViT example:

```powershell
python export_to_torchscript.py --checkpoint model.pt --out model_scripted.pt --image-size 224 --num-classes 1000 --preset vit_b_16
```

4) Start the server with the scripted file:

```powershell
$env:MODEL_PATH='model.pt'; python server.py
```

Optional labels

Create `labels.json` with one label per class (index order). Example for 2 classes:

```json
[
  "benign",
  "malignant"
]
```

Then run with the same command; the API will include `label` in the response and the UI will display it.
### Quick start cheatsheet

- If `model.pt` is a TorchScript file already: `python server.py` and open http://localhost:8000
- If `model.pt` is a state_dict: implement `build_model(...)` in `export_to_torchscript.py`, run the exporter, then start the server with `MODEL_PATH=model_scripted.pt`.


