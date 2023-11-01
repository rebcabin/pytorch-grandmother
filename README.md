To run Jupyter kernels in this repository, you will need to create a kernel for
them. Here is what I did on the Mac:

Create a virtual environment, either via conda or some more ancient method. I
called mine (unimaginatively) `venv`

In `~/Library/Jupyter/kernels/venv`, deposit this file; name it `kernel.json`

```json
{
 "argv": [
     "/Users/brian/IdeaProjects/pytorch-grandmother/projection_end_to_end/venv/bin/python",
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ],
 "display_name": "venv",
 "language": "python",
 "metadata": {
  "debugger": true
 }
}
```

Populate your environment with these packages (I know this is big; it's the
output of `pip freeze` and shows a working state.)

```
anyio==4.0.0
appnope==0.1.3
argon2-cffi==23.1.0
argon2-cffi-bindings==21.2.0
arrow==1.3.0
asttokens==2.4.1
async-lru==2.0.4
attrs==23.1.0
Babel==2.13.1
backcall==0.2.0
beautifulsoup4==4.12.2
bleach==6.1.0
certifi==2023.7.22
cffi==1.16.0
charset-normalizer==3.3.1
comm==0.1.4
contourpy==1.1.1
cycler==0.12.1
debugpy==1.8.0
decorator==5.1.1
defusedxml==0.7.1
executing==2.0.1
fastjsonschema==2.18.1
filelock==3.13.0
fonttools==4.43.1
fqdn==1.5.1
idna==3.4
ipykernel==6.26.0
ipython==8.16.1
ipywidgets==8.1.1
isoduration==20.11.0
jedi==0.19.1
Jinja2==3.1.2
joblib==1.3.2
json5==0.9.14
jsonpointer==2.4
jsonschema==4.19.1
jsonschema-specifications==2023.7.1
jupyter-events==0.8.0
jupyter-lsp==2.2.0
jupyter_client==8.5.0
jupyter_core==5.4.0
jupyter_server==2.9.1
jupyter_server_terminals==0.4.4
jupyterlab==4.0.7
jupyterlab-pygments==0.2.2
jupyterlab-widgets==3.0.9
jupyterlab_server==2.25.0
kiwisolver==1.4.5
MarkupSafe==2.1.3
matplotlib==3.8.0
matplotlib-inline==0.1.6
mistune==3.0.2
mpmath==1.3.0
nbclient==0.8.0
nbconvert==7.9.2
nbformat==5.9.2
nest-asyncio==1.5.8
networkx==3.2.1
notebook_shim==0.2.3
numpy==1.26.1
overrides==7.4.0
packaging==23.2
pandas==2.1.2
pandocfilters==1.5.0
parso==0.8.3
pexpect==4.8.0
pickleshare==0.7.5
Pillow==10.1.0
platformdirs==3.11.0
prometheus-client==0.17.1
prompt-toolkit==3.0.39
psutil==5.9.6
ptyprocess==0.7.0
pure-eval==0.2.2
pycparser==2.21
Pygments==2.16.1
pyparsing==3.1.1
python-dateutil==2.8.2
python-json-logger==2.0.7
pytz==2023.3.post1
PyYAML==6.0.1
pyzmq==25.1.1
referencing==0.30.2
requests==2.31.0
rfc3339-validator==0.1.4
rfc3986-validator==0.1.1
rpds-py==0.10.6
scikit-learn==1.3.2
scipy==1.11.3
Send2Trash==1.8.2
six==1.16.0
sniffio==1.3.0
soupsieve==2.5
stack-data==0.6.3
sympy==1.12
terminado==0.17.1
threadpoolctl==3.2.0
tinycss2==1.2.1
torch==2.0.0
torchinfo==1.8.0
torchsummary==1.5.1
torchvision==0.15.1
tornado==6.3.3
tqdm==4.66.1
traitlets==5.12.0
types-python-dateutil==2.8.19.14
typing_extensions==4.8.0
tzdata==2023.3
uri-template==1.3.0
urllib3==2.0.7
wcwidth==0.2.8
webcolors==1.13
webencodings==0.5.1
websocket-client==1.6.4
widgetsnbextension==4.0.9```