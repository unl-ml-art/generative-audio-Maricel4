# Project 2 Generative Audio

Your Name, yourcontact@unl.edu
Maricel Reinhard, maricel.reinhard@huskers.unl.edu


## Abstract

Include your abstract here. This should be one paragraph clearly describing your concept, method, and results. This should tell us what architecture/approach you used. Also describe your creative goals, and whether you were successful in achieving them. Also could describe future directions.
Hello! Elliot is helping me with the project!! :)))
I am planning on generating audio using Open AI Jukebox. I want to generate audio from BROCKHAMPTON mp3s and just from the data they already have on that music group to create a new BROCKHAMPTON SONG. I am debating whether or not to write my own lyrics or to generate new lyrics with GTP2. I think I will experiment with both and then use what I like best. I think it will be cool if time to create a short visualizer/album cover for the song as well!!!
## Model/Data
We used 25 of our favorite BROCKHAMPTON song lyrics to generate our own. We also used the songs BLEACH and BERLIN as files to upload to the code for the primed song generation.

Briefly describe the files that are included with your repository:
- trained models
- training data (or link to training data)

This was a collab notebook I stumbled upon while searching tutorials and fixes for Open On Demand. It is a much newer version of the code we used in 

## Code
#@title Select your settings and run this cell to start generating

from google.colab import drive
drive.mount('/content/gdrive')

!pip install --upgrade git+https://github.com/craftmine1000/jukebox-saveopt.git

import jukebox
import torch as t
import librosa
import os
from IPython.display import Audio
from jukebox.make_models import make_vqvae, make_prior, MODELS, make_model
from jukebox.hparams import Hyperparams, setup_hparams
from jukebox.sample import sample_single_window, _sample, \
                           sample_partial_window, upsample, \
                           load_prompts
from jukebox.utils.dist_utils import setup_dist_from_mpi
from jukebox.utils.torch_utils import empty_cache
# MPI Connect. MPI doesn't like being initialized twice, hence the following
try:
    if device is not None:
        pass
except NameError:
    rank, local_rank, device = setup_dist_from_mpi()

model = "5b_lyrics" #@param ["5b_lyrics", "1b_lyrics"]
hps = Hyperparams()
hps.sr = 44100
hps.n_samples =  2#@param {type:"integer"}
# Specifies the directory to save the sample in.
# We set this to the Google Drive mount point.
hps.name = '/content/gdrive/MyDrive/Your_folder' #@param {type:"string"}
chunk_size = 16 if model in ('5b', '5b_lyrics') else 32
gpu_info = !nvidia-smi -L
if gpu_info[0].find('Tesla T4') >= 0:
  max_batch_size = 2
  print('Tesla T4 detected, max_batch_size set to 2')
elif gpu_info[0].find('Tesla K80') >= 0:
  max_batch_size = 8
  print('Tesla K80 detected, max_batch_size set to 8')
elif gpu_info[0].find('Tesla P100') >= 0:
  max_batch_size = 3
  print('Tesla P100 detected, max_batch_size set to 3')
elif gpu_info[0].find('Tesla V100') >= 0:
  max_batch_size = 3
  print('Tesla V100 detected, max_batch_size set to 3')
else:
  max_batch_size = 3
  print('Different GPU detected, max_batch_size set to 3.')
hps.levels = 3
speed_upsampling = True #@param {type: "boolean"}
if speed_upsampling == True:
  hps.hop_fraction = [1,1,.125]
else:
  hps.hop_fraction = [.5,.5,.125]

vqvae, *priors = MODELS[model]
vqvae = make_vqvae(setup_hparams(vqvae, dict(sample_length = 1048576)), device)
top_prior = make_prior(setup_hparams(priors[-1], dict()), vqvae, device)

# The default mode of operation.
# Creates songs based on artist and genre conditioning.
mode = 'primed' #@param ["ancestral", "primed"]
if mode == 'ancestral':
  codes_file=None
  audio_file=None
  prompt_length_in_seconds=None
if mode == 'primed':
  codes_file=None
  # Specify an audio file here.
  audio_file = '/content/gdrive/My Drive/your_file.wav' #@param {type:"string"}
  # Specify how many seconds of audio to prime on.
  prompt_length_in_seconds=10 #@param {type:"integer"}

sample_length_in_seconds = 70 #@param {type:"integer"}

if os.path.exists(hps.name):
  # Identify the lowest level generated and continue from there.
  for level in [0, 1, 2]:
    data = f"{hps.name}/level_{level}/data.pth.tar"
    if os.path.isfile(data):
      codes_file = data
      if int(sample_length_in_seconds) > int(librosa.get_duration(filename=f'{hps.name}/level_2/item_0.wav')):
        mode = 'continue'
      else:
        mode = 'upsample'
      break

print('mode is now '+mode)
if mode == 'continue':
  print('Continuing from level 2')
if mode == 'upsample':
  print('Upsampling from level '+str(level))

sample_hps = Hyperparams(dict(mode=mode, codes_file=codes_file, audio_file=audio_file, prompt_length_in_seconds=prompt_length_in_seconds))

if mode == 'upsample':
  sample_length_in_seconds=int(librosa.get_duration(filename=f'{hps.name}/level_{level}/item_0.wav'))
  data = t.load(sample_hps.codes_file, map_location='cpu')
  zs = [z.cpu() for z in data['zs']]
  hps.n_samples = zs[-1].shape[0]

if mode == 'continue':
  data = t.load(sample_hps.codes_file, map_location='cpu')
  zs = [z.cpu() for z in data['zs']]
  hps.n_samples = zs[-1].shape[0]

hps.sample_length = (int(sample_length_in_seconds*hps.sr)//top_prior.raw_to_tokens)*top_prior.raw_to_tokens
assert hps.sample_length >= top_prior.n_ctx*top_prior.raw_to_tokens, f'Please choose a larger sampling rate'

# Note: Metas can contain different prompts per sample.
# By default, all samples use the same prompt.

select_artist = "the beatles" #@param {type:"string"}
select_genre = "rock" #@param {type:"string"}
metas = [dict(artist = select_artist,
            genre = select_genre,
            total_length = hps.sample_length,
            offset = 0,
            lyrics = your_lyrics, 
            ),
          ] * hps.n_samples
labels = [None, None, top_prior.labeller.get_batch_labels(metas, 'cuda')]

sampling_temperature = .98 #@param {type:"number"}

if gpu_info[0].find('Tesla T4') >= 0:
  lower_batch_size = 12
  print('Tesla T4 detected, lower_batch_size set to 12')
elif gpu_info[0].find('Tesla K80') >= 0:
  lower_batch_size = 8
  print('Tesla K80 detected, lower_batch_size set to 8')
elif gpu_info[0].find('Tesla P100') >= 0:
  lower_batch_size = 16
  print('Tesla P100 detected, lower_batch_size set to 16')
elif gpu_info[0].find('Tesla V100') >= 0:
  lower_batch_size = 16
  print('Tesla V100 detected, lower_batch_size set to 16')
else:
  lower_batch_size = 8
  print('Different GPU detected, lower_batch_size set to 8.')
lower_level_chunk_size = 32
chunk_size = 16 if model in ('5b', '5b_lyrics') else 32
sampling_kwargs = [dict(temp=.99, fp16=True, max_batch_size=lower_batch_size,
                        chunk_size=lower_level_chunk_size),
                    dict(temp=0.99, fp16=True, max_batch_size=lower_batch_size,
                         chunk_size=lower_level_chunk_size),
                    dict(temp=sampling_temperature, fp16=True, 
                         max_batch_size=max_batch_size, chunk_size=chunk_size)]

if sample_hps.mode == 'ancestral':
  zs = [t.zeros(hps.n_samples,0,dtype=t.long, device='cpu') for _ in range(len(priors))]
  zs = _sample(zs, labels, sampling_kwargs, [None, None, top_prior], [2], hps)
elif sample_hps.mode == 'upsample':
  assert sample_hps.codes_file is not None
  # Load codes.
  data = t.load(sample_hps.codes_file, map_location='cpu')
  zs = [z.cpu() for z in data['zs']]
  assert zs[-1].shape[0] == hps.n_samples, f"Expected bs = {hps.n_samples}, got {zs[-1].shape[0]}"
  del data
  print('One click upsampling!')
elif sample_hps.mode == 'primed':
  assert sample_hps.audio_file is not None
  audio_files = sample_hps.audio_file.split(',')
  duration = (int(sample_hps.prompt_length_in_seconds*hps.sr)//top_prior.raw_to_tokens)*top_prior.raw_to_tokens
  x = load_prompts(audio_files, duration, hps)
  zs = top_prior.encode(x, start_level=0, end_level=len(priors), bs_chunks=x.shape[0])
  zs = _sample(zs, labels, sampling_kwargs, [None, None, top_prior], [2], hps)
elif sample_hps.mode == 'continue':
  data = t.load(sample_hps.codes_file, map_location='cpu')
  zs = [z.cuda() for z in data['zs']]
  zs = _sample(zs, labels, sampling_kwargs, [None, None, top_prior], [2], hps)
else:
  raise ValueError(f'Unknown sample mode {sample_hps.mode}.')

# Set this False if you are on a local machine that has enough memory (this allows you to do the
# lyrics alignment visualization during the upsampling stage). For a hosted runtime, 
# we'll need to go ahead and delete the top_prior if you are using the 5b_lyrics model.
if True:
  del top_prior
  empty_cache()
  top_prior=None
upsamplers = [make_prior(setup_hparams(prior, dict()), vqvae, 'cpu') for prior in priors[:-1]]
labels[:2] = [prior.labeller.get_batch_labels(metas, 'cuda') for prior in upsamplers]

zs = upsample(zs, labels, sampling_kwargs, [*upsamplers, top_prior], hps)

Your code for generating your project:
- Python: generative_code.py
- Jupyter notebooks: generative_code.ipynb

## Results

Documentation of your results in an appropriate format, both links to files and a brief description of their contents:
- `.wav` files or `.mp4`
- `.midi` files
- musical scores
- ... some other form

## Technical Notes

Any implementation details or notes we need to repeat your work. 
- Does this code require other pip packages, software, etc?
- Does it run on some other (non-datahub) platform? (CoLab, etc.)

I had a lot of trouble with running out of memory? Cuda memory. I'd really enjoy revisiting this project with more ability to generate more clips. The code didn't work after the first pass. I had to set up my google drive to the code. That was really finicky. And everytime I ran the code I had to create a new notebook with a new name or else the code would get very confused. 

## Reference

References to any papers, techniques, repositories you used:
- Papers
- Repositories
- Blog posts
- https://colab.research.google.com/drive/1sJda9v46gNzBc7m59MP5zn63AWc-axCY?usp=sharing#scrollTo=sAdFGF-bqVMY
- https://www.youtube.com/watch?v=q21I7YO-MEs
- https://www.youtube.com/watch?v=PXxGOl-ATa4&t=0s
